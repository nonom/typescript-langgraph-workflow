import { readFileSync } from "node:fs";
import { ChatOllama } from "@langchain/ollama";
import {
  END,
  GraphNode,
  START,
  StateGraph,
  StateSchema,
} from "@langchain/langgraph";
import { z } from "zod";

const State = new StateSchema({
  log: z.string(),
  type: z.string().optional(),
  priority: z.string().optional(),
  subject: z.string().optional(),
  category: z.string().optional(),
  classification_reason: z.string().optional(),
  analysis: z.string().optional(),
  probable_cause: z.string().optional(),
  classification_confidence: z.number().optional(),
  analysis_confidence: z.number().optional(),
  recommended_next_steps: z.array(z.string()).optional(),
  validation: z.string().optional(),
  validation_passed: z.boolean().optional(),
  final_result: z.unknown().optional(),
});

const Classification = z.object({
  category: z.enum([
    "database",
    "network",
    "authentication",
    "configuration",
    "performance",
    "unknown",
  ]),
  confidence: z.number().min(0).max(100),
  reason: z.string(),
});

const Analysis = z.object({
  analysis: z.string(),
  probable_cause: z.string(),
  confidence: z.number().min(0).max(100),
  recommended_next_steps: z.array(z.string()),
});

const Validation = z.object({
  valid: z.boolean().describe(
    "True only when the diagnosis is supported by the log; false when it includes unsupported assumptions"
  ),
  feedback: z.string(),
});

// The same model performs each LLM step.
const MODEL = process.env.OLLAMA_MODEL ?? "granite3-dense:2b";
const model = new ChatOllama({
  model: MODEL,
  temperature: 0.1,
  topK: 50,
  repeatPenalty: 1.05,
});

const classifyLog: GraphNode<typeof State> = async (state) => {
    const result = await model
    .withStructuredOutput(Classification)
    .invoke([
      [
        "system",
        "Classify the log using only its evidence. Choose the category and " +
          "confidence, and give a brief reason. Use unknown when no domain is " +
          "clear. If the message is malformed or mostly noise, use unknown. " +
          "Confidence: 0-100.",
      ],
      ["human", state.log],
    ]);
  return {
    category: result.category,
    classification_confidence: result.confidence,
    classification_reason: result.reason,
  };
};

const analyseLog: GraphNode<typeof State> = async (state) => {
  const result = await model
    .withStructuredOutput(Analysis)
    .invoke([
      [
        "system",
        "Analyse the log. Give the probable cause and next steps supported " +
          "by the log. Confidence: 0-100.",
      ],
      ["human", state.log],
    ]);
  return {
    analysis: result.analysis,
    probable_cause: result.probable_cause,
    analysis_confidence: result.confidence,
    recommended_next_steps: result.recommended_next_steps,
  };
};

const validateAnalysis: GraphNode<typeof State> = async (state) => {
  const result = await model
    .withStructuredOutput(Validation)
    .invoke([
      [
        "system",
        "Review the diagnosis against the log. Check that the cause and next " +
          "steps are supported by the evidence.",
      ],
      [
        "human",
        JSON.stringify({
          log: state.log,
          category: state.category,
          analysis: state.analysis,
          probable_cause: state.probable_cause,
          confidence: state.analysis_confidence,
          recommended_next_steps: state.recommended_next_steps,
        }),
      ],
    ]);
  return {
    validation: result.feedback,
    validation_passed: result.valid,
  };
};

const finalResponse: GraphNode<typeof State> = (state) => ({
  final_result: {
    category: state.category,
    probable_cause: state.probable_cause,
    confidence: state.analysis_confidence,
    recommended_next_steps: state.recommended_next_steps,
  },
});

const rejectedResponse: GraphNode<typeof State> = (state) => ({
  final_result: {
    status: "rejected",
    reason: state.validation ?? state.classification_reason ?? "The log could not be classified.",
  },
});

const routeAfterClassification = (state: typeof State.State) =>
  state.category === "unknown" ? "rejected_response" : "analyse_log";

const routeAfterValidation = (state: typeof State.State) =>
  state.validation_passed ? "final_response" : "rejected_response";

const graph = new StateGraph(State)
  .addNode("classify_log", classifyLog)
  .addNode("analyse_log", analyseLog)
  .addNode("validate_analysis", validateAnalysis)
  .addNode("final_response", finalResponse)
  .addNode("rejected_response", rejectedResponse)
  .addEdge(START, "classify_log")
  .addConditionalEdges("classify_log", routeAfterClassification, {
    analyse_log: "analyse_log",
    rejected_response: "rejected_response",
  })
  .addEdge("analyse_log", "validate_analysis")
  .addConditionalEdges("validate_analysis", routeAfterValidation, {
    final_response: "final_response",
    rejected_response: "rejected_response",
  })
  .addEdge("final_response", END)
  .addEdge("rejected_response", END)
  .compile();

function readLog(fileName?: string, input?: string): string {
  const content = input ?? (fileName
    ? readFileSync(fileName, "utf8")
    : !process.stdin.isTTY
      ? readFileSync(0, "utf8")
      : (() => {
          throw new Error("Usage: npm run start -- example.log --verbose");
        })());

  const line = content.split(/\r?\n/).find((value) => value.trim());
  if (!line) throw new Error("Expected one non-empty log line.");
  return line.trim();
}

const args = process.argv.slice(2);
const inputIndex = args.indexOf("--input");
const input = inputIndex >= 0 ? args[inputIndex + 1] : undefined;
if (inputIndex >= 0 && !input) throw new Error("--input requires a log line.");
const fileName = args.find(
  (value, index) => !value.startsWith("-") && index !== inputIndex + 1
);
const verbose = args.includes("--verbose");
const initialState = { log: readLog(fileName, input) };

if (verbose) {
  const green = process.stdout.isTTY ? "\x1b[32m" : "";
  const reset = green ? "\x1b[0m" : "";
  console.log(`\n${green}Log input:${reset} ${initialState.log}`);

  const nodeModels: Record<string, string> = {
    classify_log: MODEL,
    analyse_log: MODEL,
    validate_analysis: MODEL,
    final_response: "TypeScript",
    rejected_response: "TypeScript",
  };

  for await (const update of await graph.stream(initialState, {
    streamMode: "updates",
  })) {
    for (const [node, output] of Object.entries(update)) {
      console.log(`\n${green}[${node} · ${nodeModels[node]}]${reset}`);
      console.log(JSON.stringify(output, null, 2));
    }
  }
} else {
  const result = await graph.invoke(initialState);
  console.log(JSON.stringify(result.final_result, null, 2));
}
