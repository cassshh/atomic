import type { ClassifierFunction, ClassifierOptions, ImageContent } from "../types.ts";
import { isRecord } from "./classifier-shared.ts";
import { classifySystemOne, type SystemOneTransport } from "./system-one-shared.ts";

const LABEL = "Gemini Decisions";

/** Gemini's yes/no criteria name their descriptions; System One's are `true` and `false`. */
function wireQuestion(question: unknown): unknown {
	if (!isRecord(question) || question.type !== "noul" || !isRecord(question.criteria)) return question;
	const { true: yes, false: no } = question.criteria;
	return {
		...question,
		criteria: {
			...(typeof yes === "string" && yes ? { true_description: yes } : {}),
			...(typeof no === "string" && no ? { false_description: no } : {}),
		},
	};
}

/** The shared input: the state as text, followed by any images as content blocks. */
function wireInput(state: unknown, images: ImageContent[] | undefined): unknown {
	const text = typeof state === "string" ? state : JSON.stringify(state);
	if (!images?.length) return text;
	return [
		{ type: "text", text },
		...images.map((image) => ({
			type: "image",
			data: image.data.replace(/^data:[^,]*,/u, ""),
			mime_type: image.mimeType,
		})),
	];
}

function tokens(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * Gemini Decisions (`gemini-decision-1`) on the Gemini Developer API:
 * `POST {base}/decisions` with `{ model, input, questions }`, keyed by `x-goog-api-key`.
 * It speaks System One's question and answer shapes (`noul`, `choice`, `score`, keyed by question
 * ID), so only the envelope differs: the state goes in `input` as text or content blocks, and usage
 * is reported as `total_input_tokens`/`total_output_tokens`. Question IDs must match
 * `[A-Za-z0-9_-]{1,64}`.
 * https://aistudio.google.com/docs/decisions-api
 */
const transport: SystemOneTransport = {
	api: "google-decisions",
	label: LABEL,
	detectCompat: () => ({ imageInput: "images", sendImageOptInHeader: false }),
	url: (model) => new URL("decisions", `${model.baseUrl.replace(/\/+$/u, "")}/`),
	ownsImages: true,
	authHeaders: (apiKey) => ({ "x-goog-api-key": apiKey }),
	payload: (model, request, images) => ({
		model: model.id,
		input: wireInput(request.state, images),
		questions: Object.fromEntries(Object.entries(request.questions).map(([id, q]) => [id, wireQuestion(q)])),
	}),
	output: (body) => {
		if (!isRecord(body)) throw new Error(`${LABEL} returned an unexpected response`);
		const usage = isRecord(body.usage) ? body.usage : undefined;
		return {
			answers: body.answers,
			usage: usage && {
				input_tokens: tokens(usage.total_input_tokens),
				output_tokens: tokens(usage.total_output_tokens),
			},
		};
	},
};

/** Gemini Decisions classification with public `bool` values mapped to wire-level `noul`. */
export const classify: ClassifierFunction<ClassifierOptions> = (model, context, options) =>
	classifySystemOne(transport, model, context, options);
