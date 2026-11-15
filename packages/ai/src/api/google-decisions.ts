import type {
	ClassifierAnswer,
	ClassifierContext,
	ClassifierFunction,
	ClassifierOptions,
	ClassifierQuestion,
	ImageContent,
} from "../types.ts";
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

/** Gemini's rule for a question ID. */
const QUESTION_ID = /^[A-Za-z0-9_-]{1,64}$/u;

/** Gemini's rule for a choice option key: 1 to 128 characters, at most 512 bytes, no edge spaces or controls. */
function validOption(key: string): boolean {
	return (
		key.length >= 1 &&
		key.length <= 128 &&
		new TextEncoder().encode(key).length <= 512 &&
		key === key.trim() &&
		!/\p{Cc}/u.test(key)
	);
}

/** Stand-in names that keep their order when Gemini sorts them, as it does option keys. */
function standIn(prefix: string, index: number): string {
	return `${prefix}${String(index).padStart(3, "0")}`;
}

/**
 * Questions with every ID and option key Gemini accepts, and the way back to the caller's names. A
 * schema property such as `review.status` becomes a question ID, so names Gemini refuses are sent
 * as stand-ins and the answers are returned under the caller's own names.
 */
function wireNames(questions: Record<string, ClassifierQuestion>) {
	const ids = Object.keys(questions);
	const renameIds = ids.some((id) => !QUESTION_ID.test(id));
	const wireId = new Map(ids.map((id, index) => [id, renameIds ? standIn("q", index) : id]));
	const options = new Map<string, Map<string, string>>();
	const wire: Record<string, ClassifierQuestion> = {};
	for (const [id, question] of Object.entries(questions)) {
		let sent = question;
		if (question.type === "choice") {
			const keys = Object.keys(question.criteria);
			if (keys.some((key) => !validOption(key))) {
				const names = new Map(keys.map((key, index) => [key, standIn("o", index)]));
				options.set(id, names);
				sent = {
					...question,
					criteria: Object.fromEntries(keys.map((key) => [names.get(key)!, question.criteria[key]!])),
				};
			}
		}
		wire[wireId.get(id)!] = sent;
	}
	const renamed = renameIds || options.size > 0;
	const restore = (answers: Record<string, ClassifierAnswer>): Record<string, ClassifierAnswer> =>
		Object.fromEntries(
			ids.flatMap((id) => {
				const answer = answers[wireId.get(id)!];
				if (!answer) return [];
				const names = options.get(id);
				if (answer.type !== "choice" || !names) return [[id, answer]];
				const back = new Map([...names].map(([key, name]) => [name, key]));
				return [
					[
						id,
						{
							...answer,
							choice: back.get(answer.choice) ?? answer.choice,
							probabilities: Object.fromEntries(
								Object.entries(answer.probabilities).map(([name, p]) => [back.get(name) ?? name, p]),
							),
						},
					],
				];
			}),
		);
	return { renamed, wire, restore };
}

/** Gemini Decisions classification with public `bool` values mapped to wire-level `noul`. */
export const classify: ClassifierFunction<ClassifierOptions> = async (model, context, options) => {
	const names = wireNames(context.questions);
	if (!names.renamed) return classifySystemOne(transport, model, context, options);
	const sent: ClassifierContext = { ...context, questions: names.wire };
	const result = await classifySystemOne(transport, model, sent, options);
	return { ...result, answers: names.restore(result.answers) };
};
