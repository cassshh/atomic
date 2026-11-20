import { describe, expect, it, vi } from "vitest";
import { classify } from "../src/api/google-decisions.ts";
import { builtinModels } from "../src/providers/all.js";
import type { ClassifierContext, ClassifierModel, ImageContent, KnownProvider } from "../src/types.js";

const png: ImageContent = { type: "image", data: "data:image/png;base64,iVBORw0KGgo=", mimeType: "image/png" };

const model: ClassifierModel<"google-decisions"> = {
	type: "classifier",
	id: "gemini-decision-1",
	name: "Gemini Decision 1",
	api: "google-decisions",
	provider: "google-decisions",
	baseUrl: "https://generativelanguage.googleapis.com/v1beta",
	input: ["text", "image"],
	cost: { input: 0.1, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 262144,
};

const context: ClassifierContext = {
	state: { text: "The deployment succeeded, thank you." },
	questions: {
		category: {
			type: "choice",
			instructions: "Classify the message",
			criteria: { success: "Successful", failure: "" },
		},
		satisfaction: {
			type: "score",
			instructions: "Score satisfaction",
			criteria: ["low", "neutral", "high"],
		},
		approved: {
			type: "bool",
			instructions: "Does the user approve?",
			criteria: { true: "Approval", false: "No approval" },
		},
	},
};

const wireAnswers = {
	category: {
		type: "choice",
		choice: "success",
		probabilities: { success: 0.9, failure: 0.1 },
		confidence: 0.9,
	},
	satisfaction: {
		type: "score",
		score: 1.8,
		legend: { "0": "low", "1": "neutral", "2": "high" },
		probabilities: { "0": 0.05, "1": 0.1, "2": 0.85 },
		confidence: 0.85,
	},
	approved: { type: "noul", noul: 0.95, confidence: 0.95 },
};

interface Captured {
	url: string;
	headers: Headers;
	body: Record<string, unknown>;
}

function capturingFetch(response: () => Response) {
	const requests: Captured[] = [];
	const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
		requests.push({
			url: String(input),
			headers: new Headers(init?.headers),
			body: JSON.parse(String(init?.body)) as Record<string, unknown>,
		});
		return response();
	});
	return { fetch, requests };
}

const ok = () =>
	Response.json({
		answers: wireAnswers,
		usage: {
			total_input_tokens: 120,
			input_tokens_by_modality: [{ modality: "text", tokens: 120 }],
			total_output_tokens: 0,
			total_tokens: 120,
		},
	});

describe("Gemini Decisions", () => {
	it("posts the Gemini envelope with the key in x-goog-api-key", async () => {
		const { fetch, requests } = capturingFetch(ok);

		const result = await classify(model, context, { apiKey: "secret", fetch });

		expect(result.stopReason).toBe("stop");
		expect(requests).toHaveLength(1);
		const [request] = requests;
		expect(request?.url).toBe("https://generativelanguage.googleapis.com/v1beta/decisions");
		expect(request?.headers.get("x-goog-api-key")).toBe("secret");
		expect(request?.headers.get("authorization")).toBeNull();
		expect(request?.body).toEqual({
			model: "gemini-decision-1",
			input: JSON.stringify(context.state),
			questions: {
				category: {
					type: "choice",
					instructions: "Classify the message",
					criteria: { success: "Successful", failure: "" },
				},
				satisfaction: {
					type: "score",
					instructions: "Score satisfaction",
					criteria: ["low", "neutral", "high"],
				},
				approved: {
					type: "noul",
					instructions: "Does the user approve?",
					criteria: { true_description: "Approval", false_description: "No approval" },
				},
			},
		});
	});

	it("maps noul, choice and score answers and reads Gemini usage", async () => {
		const { fetch } = capturingFetch(ok);

		const result = await classify(model, context, { apiKey: "secret", fetch });

		expect(result.answers).toEqual({
			category: {
				type: "choice",
				choice: "success",
				probabilities: { success: 0.9, failure: 0.1 },
				confidence: 0.9,
			},
			satisfaction: { type: "score", score: 1.8, confidence: 0.85 },
			approved: { type: "bool", probability: 0.95 },
		});
		expect(result.usage).toMatchObject({ input: 120, output: 0, totalTokens: 120 });
		expect(result.usage?.cost.total).toBeCloseTo(0.000012);
	});

	it("sends a string state as is and omits empty yes/no descriptions", async () => {
		const { fetch, requests } = capturingFetch(() =>
			Response.json({ answers: { done: { type: "noul", noul: 0.2, confidence: 0.8 } } }),
		);

		const result = await classify(
			model,
			{
				state: "Order arrived with a cracked lid." as unknown as ClassifierContext["state"],
				questions: { done: { type: "bool", instructions: "Is it damaged?", criteria: { true: "", false: "" } } },
			},
			{ apiKey: "secret", fetch },
		);

		expect(result.stopReason).toBe("stop");
		expect(requests[0]?.body.input).toBe("Order arrived with a cracked lid.");
		expect(requests[0]?.body.questions).toEqual({
			done: { type: "noul", instructions: "Is it damaged?", criteria: {} },
		});
	});

	it("sends images as content blocks after the state", async () => {
		const { fetch, requests } = capturingFetch(ok);

		await classify(model, { ...context, images: [png] }, { apiKey: "secret", fetch });

		expect(requests[0]?.body.input).toEqual([
			{ type: "text", text: JSON.stringify(context.state) },
			{ type: "image", data: "iVBORw0KGgo=", mime_type: "image/png" },
		]);
		expect(requests[0]?.headers.get("typesafe-danger-zone-images")).toBeNull();
	});

	it("reports a safety refusal as an error with its status", async () => {
		const { fetch } = capturingFetch(() =>
			Response.json({ error: { code: "safety", message: "Input was blocked for safety." } }, { status: 400 }),
		);

		const result = await classify(model, context, { apiKey: "secret", fetch, maxRetries: 0 });

		expect(result.stopReason).toBe("error");
		expect(result.answers).toEqual({});
		expect(result.errorMessage).toContain("400");
	});

	it("rejects an answer for a choice the question does not offer", async () => {
		const { fetch } = capturingFetch(() =>
			Response.json({ answers: { ...wireAnswers, category: { ...wireAnswers.category, choice: "other" } } }),
		);

		const result = await classify(model, context, { apiKey: "secret", fetch });

		expect(result.stopReason).toBe("error");
	});

	it("sends names Gemini refuses as stand-ins and answers under the caller's names", async () => {
		const long = "a".repeat(65);
		const { fetch, requests } = capturingFetch(() =>
			Response.json({
				answers: {
					q000: {
						type: "choice",
						choice: "o001",
						probabilities: { o000: 0.2, o001: 0.8 },
						confidence: 0.8,
					},
					q001: { type: "noul", noul: 0.3, confidence: 0.7 },
				},
			}),
		);

		const result = await classify(
			model,
			{
				state: { text: "Looks good to me" },
				questions: {
					"review.status": {
						type: "choice",
						instructions: "Status",
						criteria: { " pending": "Waiting", approved: "Approved" },
					},
					[long]: { type: "bool", instructions: "Merged?", criteria: { true: "", false: "" } },
				},
			},
			{ apiKey: "secret", fetch },
		);

		expect(requests[0]?.body.questions).toEqual({
			q000: {
				type: "choice",
				instructions: "Status",
				criteria: { o000: "pending: Waiting", o001: "approved: Approved" },
			},
			q001: { type: "noul", instructions: "Merged?", criteria: {} },
		});
		expect(result.stopReason).toBe("stop");
		expect(result.answers).toEqual({
			"review.status": {
				type: "choice",
				choice: "approved",
				probabilities: { " pending": 0.2, approved: 0.8 },
				confidence: 0.8,
			},
			[long]: { type: "bool", probability: 0.3 },
		});
	});

	it("keeps a label-only option's name, and a __proto__ question, when options are renamed", async () => {
		const { fetch, requests } = capturingFetch(() =>
			Response.json({
				// From entries, as JSON.parse makes it: a literal `__proto__:` would set the prototype.
				answers: Object.fromEntries([
					["__proto__", { type: "noul", noul: 0.4, confidence: 0.6 }],
					["status", { type: "choice", choice: "o000", probabilities: { o000: 0.9, o001: 0.1 }, confidence: 0.9 }],
				]),
			}),
		);
		const questions = Object.fromEntries([
			["__proto__", { type: "bool", instructions: "Odd name?", criteria: { true: "", false: "" } }],
			["status", { type: "choice", instructions: "Status", criteria: { " open": "", closed: "" } }],
		]) as ClassifierContext["questions"];

		const result = await classify(model, { state: { text: "x" }, questions }, { apiKey: "secret", fetch });

		const sent = requests[0]?.body.questions as Record<string, unknown>;
		expect(Object.keys(sent)).toEqual(["__proto__", "status"]);
		expect(sent.status).toEqual({
			type: "choice",
			instructions: "Status",
			criteria: { o000: "open", o001: "closed" },
		});
		expect(result.stopReason).toBe("stop");
		expect(Object.keys(result.answers)).toEqual(["__proto__", "status"]);
		expect(result.answers.status).toMatchObject({ choice: " open" });
	});

	it("answers nothing for a renamed request that failed, whatever its question IDs", async () => {
		const { fetch } = capturingFetch(() => Response.json({ error: { code: "safety" } }, { status: 400 }));
		const questions = Object.fromEntries([
			["constructor", { type: "bool", instructions: "x", criteria: { true: "", false: "" } }],
			["toString", { type: "choice", instructions: "y", criteria: { " a": "", b: "" } }],
		]) as ClassifierContext["questions"];

		const result = await classify(model, { state: { text: "x" }, questions }, { apiKey: "secret", fetch, maxRetries: 0 });

		expect(result.stopReason).toBe("error");
		expect(Object.keys(result.answers)).toEqual([]);
	});

	it("sends names Gemini accepts unchanged", async () => {
		const { fetch, requests } = capturingFetch(ok);

		await classify(model, context, { apiKey: "secret", fetch });

		expect(Object.keys(requests[0]?.body.questions as object)).toEqual(["category", "satisfaction", "approved"]);
	});

	it("is a built-in classifier keyed by GEMINI_API_KEY", async () => {
		const provider: KnownProvider = "google-decisions";
		expect(provider).toBe("google-decisions");
		const models = builtinModels({
			authContext: {
				env: async (name) => (name === "GEMINI_API_KEY" ? "gemini-key" : undefined),
				fileExists: async () => false,
			},
		});

		const builtin = models.getModelOfType("classifier", "google-decisions", "gemini-decision-1");
		expect(builtin).toMatchObject({ api: "google-decisions", contextWindow: 262144 });
		expect((await models.getAvailableOfType("classifier", "google-decisions")).map((entry) => entry.id)).toEqual([
			"gemini-decision-1",
		]);
		expect((await models.getAuth("google-decisions"))?.source).toBe("GEMINI_API_KEY");
	});
});
