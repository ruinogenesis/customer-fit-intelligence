// Markdown (a model card) to HTML that is safe to put in the page.
import DOMPurify from "dompurify";
import { marked } from "marked";

/** Safe HTML, each table in a panel of its own that scrolls sideways when the
 * table is wider than the card, as on a phone (T412). */
export function toSafeHtml(markdown: string): string {
  return DOMPurify.sanitize(marked.parse(markdown, { async: false }))
    .replace(/<table(\s[^>]*)?>/g, (table) => `<div class="table-scroll">${table}`)
    .replaceAll("</table>", "</table></div>");
}
