Passages were retrieved for one sub-query. Decide whether the answer is **in**
them. Topical is not sufficient: the only question is whether one passage states
the specific figure, rate, band or rule asked for.

- `sufficient` — true only if you can point at the passage containing the answer.
- `answer_chunk_id` — that passage's `chunk_id`, copied exactly. **If you cannot
  name one, `sufficient` is false.** There is no third option.
- `reason` — at most 15 words.
- `missing` — at most 10 words: the table or figure that was absent.
- `suggested_query` — null when sufficient; otherwise the query to try instead.

Reject a figure from a different year or country than the sub-query asks for, even
if it is the only number present.

The trap this exists for: "the top tax bracket percentage for a single person"
retrieves prose about filing statuses — full of the word "single", containing no
rate schedule. That is insufficient, and saying so is the only thing that recovers
it.

Writing `suggested_query`: you have just read the document, so use **its**
vocabulary, not the user's — if the passages say "Tax Rate Schedules" and
"Schedule X", search that, not "tax bracket". Name the table or section you want,
and keep the qualifiers that identify the row: filing status, an age condition,
"taxable income over". Drop conversational framing entirely — the query only has
to find the table.

**Include the year if the document prints it in the table's own title.** These
documents do ("2024 Tax Rate Schedules"), and matching a title lexically is what
pulls the right page to rank 1. Do not add the country — that is a hard filter and
adds nothing to the text.

Be specific about *which* table. "Tax Table" and "Tax Rate Schedules" are two
different things in this document: the Tax Table lists tax due for incomes below a
cutoff, and only the Rate Schedules state the bracket percentages. Asking for the
wrong one retrieves a real table that cannot answer the question.
