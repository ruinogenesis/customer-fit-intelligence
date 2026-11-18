// Markdown (a model card) to HTML that is safe to put in the page.
import DOMPurify from "dompurify";
import { marked } from "marked";

export function toSafeHtml(markdown: string): string {
  return DOMPurify.sanitize(marked.parse(markdown, { async: false }));
}
