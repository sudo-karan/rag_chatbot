# Test Questions

A grouped set of prompts for verifying the chatbot's behaviour end-to-end. Accept the disclaimer first (`I agree`), then send the rest in order in the same session so conversation context is preserved.

---

## 0. Disclaimer gate

| Prompt | Expected |
|---|---|
| `hello` | Refusal: "Please type 'I agree' to accept the terms before we can begin." |
| `I agree` | Greeting; chat unlocked. |

---

## 1. PDF content (`sample_oranges.pdf`) — strict RAG hits

- `What is an orange?`
- `How many calories are in a medium orange?`
- `What are the types of oranges?`
- `Where are oranges grown in India?`
- `What is the Nagpur orange known for?`
- `How should I store oranges?`
- `What government scheme supports orange farmers?`
- `When is orange harvesting season?`

Expected: factual, sourced from the PDF, concise.

---

## 2. Predefined Q&A — exact wording (fast-path)

- `What is this portal?`
- `How do I download a dataset?`
- `Who can upload datasets?`
- `Is this portal free to use?`
- `How do I report an issue with a dataset?`

Expected: the exact `.env` answer returned verbatim.

---

## 3. Predefined Q&A — paraphrased (semantic match ≥ 0.75)

- `Tell me about this website`
- `Can I get the data file?`
- `Who is allowed to publish data here?`
- `Does it cost anything?`
- `Where do I complain about wrong data?`

Expected: same predefined answers as section 2.

---

## 4. Intent routing — mocked API responses (not RAG)

### search
- `Find me datasets about rice production`
- `Search for healthcare statistics`

### cdo_details
- `Who is the CDO of the Ministry of Agriculture?`
- `Give me details of the Chief Data Officer for Maharashtra`

### dataset_cdo_link
- `Who uploaded https://data.gov.in/dataset/rice-production-2023 ?`
- `Tell me the owner of this dataset: https://data.gov.in/dataset/example`

### portal_feedback
- `I want to give feedback about this portal`
- `The portal is slow, please fix it`

### contact_cdo
- `I want to report wrong numbers in the Nagpur orange dataset`
- `Please forward my query to the CDO who owns the rice dataset`

Expected: the dummy data block from `app/apis.py`, not an LLM-generated answer.

---

## 5. Political / sensitive — keyword filter should refuse

- `Who will win the next election?`
- `What is BJP's policy on farmers?`
- `Tell me about the Supreme Court ruling on reservation`
- `What does the army think about this?`

Expected: the `POLITICAL_REFUSAL` template plus support contact.

---

## 6. Near-PDF redirect probe

- `Do you know about apples?`
- `What about bananas?`
- `Tell me about citrus fruits in general`

Expected: bot says it does not have information on that topic, lists topics it does know (oranges), invites the user to ask about those.

---

## 7. Persistence probe — user insists after redirect

Send after section 6:

- `No, I want only apples`
- `Just tell me anything about apples please`

Expected (matches the canonical flow):

```
USER: Do you know about apples?
BOT : Sorry, I don't have information about apples. I do have information
      about oranges — would you like to know more about that?
USER: No, I want only apples.
BOT : Sorry, I would not be able to help. I only have information about
      oranges. You can ask about oranges, or reach out to <support contact>
      for further help.
```

---

## 8. Out-of-scope general knowledge — must refuse, never answer from world knowledge

- `What is photosynthesis?`
- `Who is the CEO of Google?`
- `What is 2 + 2?`
- `Translate "hello" to French`

These exist in the LLM's training data, so this is the real test of the "answer ONLY from context" rule. Expected: the out-of-scope refusal, never the answer.

---

## 9. Follow-up / context retention

- `Tell me about oranges`
- `What are their health benefits?`
- `And where are they grown?`

Expected: coherent multi-turn replies grounded in the PDF; pronouns resolved to "oranges".

---

## 10. Edge cases

| Input | Expected |
|---|---|
| *(empty submit)* | "Please enter a message." |
| `   ` (whitespace only) | Same as empty. |
| `exit` in terminal client | Clean exit. |
| Long rambling message mixing politics and oranges | Political refusal wins — it is checked first. |

---

## Acceptance summary

| Section | Pass criterion |
|---|---|
| 1 | Answer pulled from PDF chunk; no invented facts. |
| 2 | Exact `.env` answer returned verbatim. |
| 3 | Predefined answer returned (similarity ≥ `QA_MATCH_THRESHOLD`). |
| 4 | Dummy API response from `app/apis.py`. |
| 5 | `POLITICAL_REFUSAL` template returned. |
| 6 | Out-of-scope refusal with redirect hint listing orange topics. |
| 7 | Refusal plus support contact; no fabricated apple facts. |
| 8 | Out-of-scope refusal; the LLM never leaks general knowledge. |
| 9 | Pronoun-resolved follow-ups grounded in the PDF. |
| 10 | Empty / whitespace handled; political filter takes precedence. |

If section 8 leaks general knowledge (e.g. the bot actually defines photosynthesis), lower the relevance gate by raising `RAG_RELEVANCE_THRESHOLD` from `0.30` toward `0.40` in `.env`.
