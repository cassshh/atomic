import { googleDecisionsApi } from "../api/google-decisions.lazy.ts";
import { sharedApiKeyAuth } from "../auth/helpers.ts";
import { createProvider, type Provider } from "../models.ts";
import type { ClassifierModel } from "../types.ts";
import { GOOGLE_DECISIONS_CLASSIFIER_MODELS } from "./google-decisions.models.ts";

export function googleDecisionsProvider(): Provider {
	return createProvider({
		id: "google-decisions",
		name: "Google Decisions",
		baseUrl: "https://generativelanguage.googleapis.com/v1beta",
		auth: {
			apiKey: sharedApiKeyAuth("Gemini API key", ["GEMINI_API_KEY"], ["google"]),
		},
		models: Object.values<ClassifierModel<"google-decisions">>(GOOGLE_DECISIONS_CLASSIFIER_MODELS),
		classifiers: {
			"google-decisions": googleDecisionsApi(),
		},
	});
}
