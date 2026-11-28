# Firestore documents

Made from `fitcore.documents` (T301) by
`python -m fitcore.documents > fitcore/DOCUMENTS.md`; do not edit by hand.
The API checks every document against these schemas before it writes it.
Timestamps carry a timezone and are kept in UTC; money is whole satang.

## `events/{event_id}`

One per event received, written by the API: an event as received, and what it got back.

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `event` | map |  | The event as received and validated (fitcore.schemas) |
| `result` | map |  | What POST /events answered |
| `result.event_id` | UUID, as text |  | The event's id, as text |
| `result.customer_id` | id |  | The event's customer |
| `result.duplicate` | boolean |  | True if this event_id was already received |
| `result.order_id` | id | or null | The order this event re-scored; null if none |
| `result.scores` | map | or null | That order's new scores; null if none changed |
| `result.scores.respect` | number | 0 to 1 | Respect |
| `result.scores.love` | number | 0 to 1 | Love |
| `result.scores.convenience` | number | 0 to 1 | Convenience value |
| `result.scores.trust` | number | 0 to 1 | Trust |
| `result.scores.fit` | number | 0 to 1 | Overall fit |
| `result.scores.churn_risk` | number | 0 to 1 | Chance of no order in the next 30 days |
| `result.scores.quadrant` | string | one of `lovemark`, `brand`, `fad`, `product` | Love-matrix quadrant, from the model's lines (D5) |
| `result.drivers` | list of strings | or null | The inputs that moved that order's fit most, biggest first (at most 3); null if no score changed |
| `result.model_version` | string |  | The model that scored it |
| `expires_at` | timestamp |  | When Firestore's TTL policy deletes it: 30 days after the API received it (T305); BigQuery keeps every event |

## `customers/{customer_id}`

One per customer, written by the API: a customer's expectations and latest scores.

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `customer_id` | id |  | The customer |
| `expectations` | map | keys x2..x8 | Expectation per input, x2..x8 (x1 is judged against the promise) |
| `expectations.{key}.mean` | number |  | Exponentially weighted mean, in the input's units |
| `expectations.{key}.weight` | number | 0 or more | Effective number of observations behind it |
| `expectations.{key}.last_seen` | timestamp | or null | When the last observation came in; null before the first |
| `latest_order_id` | id | or null | The latest order placed; null before the first |
| `latest_placed_at` | timestamp | or null | When it was placed; null before the first order |
| `scores` | map | or null | The latest order's latents; null before the first scoring |
| `scores.respect` | number | 0 to 1 | Respect |
| `scores.love` | number | 0 to 1 | Love |
| `scores.convenience` | number | 0 to 1 | Convenience value |
| `scores.trust` | number | 0 to 1 | Trust |
| `scores.fit` | number | 0 to 1 | Overall fit |
| `churn_risk` | number | 0 to 1; or null | The latest order's churn risk; null before the first |
| `quadrant` | string | one of `lovemark`, `brand`, `fad`, `product`; or null | The latest order's quadrant; null before the first |
| `model_version` | string |  | The model behind the scores |
| `consent` | map | or null | Their latest marketing consent (T610); null when never asked, which counts as no consent |
| `consent.marketing` | boolean |  | True if given, false if withdrawn |
| `consent.at` | timestamp |  | When the customer gave or withdrew it |
| `consent.source` | string | one of `sign_up`, `settings`, `support` | Where they did |

## `customers/{customer_id}/scores/{event_id}`

One per scoring of an order, written by the API: one scoring of one order.

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `event_id` | UUID, as text |  | The event that scored it, as text |
| `order_id` | id |  | The order scored |
| `at` | timestamp |  | When that event occurred |
| `scores` | map |  | The order's latents |
| `scores.respect` | number | 0 to 1 | Respect |
| `scores.love` | number | 0 to 1 | Love |
| `scores.convenience` | number | 0 to 1 | Convenience value |
| `scores.trust` | number | 0 to 1 | Trust |
| `scores.fit` | number | 0 to 1 | Overall fit |
| `churn_risk` | number | 0 to 1 | The order's churn risk |
| `quadrant` | string | one of `lovemark`, `brand`, `fad`, `product` | The order's quadrant |
| `contributions` | map | keys respect, love, convenience, trust, fit, churn_risk | Each input's share of how far each score sits from an order exactly as expected (every gap 0); a score's eight add up to that distance (T115) |
| `contributions.{key}.x1` | number | -1 to 1 | Speed vs promise |
| `contributions.{key}.x2` | number | -1 to 1 | App ease |
| `contributions.{key}.x3` | number | -1 to 1 | Fee load |
| `contributions.{key}.x4` | number | -1 to 1 | Order fidelity |
| `contributions.{key}.x5` | number | -1 to 1 | Courier conduct |
| `contributions.{key}.x6` | number | -1 to 1 | Food and selection |
| `contributions.{key}.x7` | number | -1 to 1 | Personalization and delight |
| `contributions.{key}.x8` | number | -1 to 1 | Recovery handling |
| `drivers` | list of strings |  | The inputs that moved fit most, up or down, biggest first (at most 3) |
| `model_version` | string |  | The model that scored it |

## `orders/{order_id}`

One per order, written by the API: an order's record and its gaps so far.

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `order_id` | id |  | The order |
| `customer_id` | id |  | Who placed it |
| `restaurant_id` | id |  | Where from |
| `placed_at` | timestamp |  | When it was placed |
| `promised_eta` | timestamp |  | When it was promised |
| `basket_value` | integer | 0 or more | Basket value, in satang |
| `fees` | integer | 0 or more | Fees, in satang |
| `promo_used` | boolean |  | True if it used a promo |
| `courier_id` | id | or null | Who delivered it; null until delivered |
| `delivered_at` | timestamp | or null | When it was delivered; null until then |
| `issues` | list of maps |  | Every order_issue, in the order received |
| `issues[].issue_type` | string | one of `missing`, `wrong`, `cold`, `late` | What went wrong |
| `issues[].at` | timestamp |  | When it was reported |
| `ratings` | map | keys food, courier, accuracy, overall | The latest rating of each kind, 1 to 5 |
| `gaps` | map | keys x1..x8 | The inputs this order has measured, in model units |

## `rules/{rule_id}`

One per automation rule, written by the API, for admins: an automation rule (T501 sets condition and action).

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `rule_id` | id |  | The rule |
| `name` | string |  | What staff call it |
| `kind` | string | one of `service`, `marketing` | service rules skip the marketing consent and frequency cap |
| `status` | string | one of `dry_run`, `test`, `live`, `off` | dry_run logs only; test and live act; off is the kill switch |
| `holdout_share` | number | 0 to 1 | Share of customers held out: 0.5 in test, 0.1 when live |
| `condition` | map |  | When it fires; its form is set by T501 |
| `action` | map |  | What it sends; its form is set by T501 |
| `caps` | map |  | Its own limits; their form is set by T502 |
| `updated_at` | timestamp |  | When it last changed |

## `actions/{action_id}`

One per rule decision, written by Cloud Functions: one decision of one rule, holdouts included.

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `action_id` | id |  | The action |
| `rule_id` | id |  | The rule that decided it |
| `customer_id` | id |  | For whom |
| `arm` | string | one of `treatment`, `holdout` | treatment acts; holdout is only logged |
| `channel` | string | one of `push`, `in_app`, `email`, `voucher` | Where it goes |
| `dry_run` | boolean |  | True if the rule was in dry_run, so nothing was sent |
| `decided_at` | timestamp |  | When the rule decided |
| `outcome` | string | one of `delivered`, `opened`, `clicked`, `redeemed`, `failed`; or null | The furthest it got; null until known |

## `models/{version}`

One per model version, written by the training pipeline: a model version in the registry (T310).

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `version` | id |  | The model version |
| `status` | string | one of `candidate`, `live`, `retired` | Only one version is live at a time |
| `path` | string |  | Its model.json in Cloud Storage; model_card.md sits next to it |
| `sha256` | string |  | model.json's SHA-256, checked before it is used |
| `metrics` | map |  | Held-out metrics by name, such as churn_auc |
| `created_at` | timestamp |  | When it was trained |
| `promoted_at` | timestamp | or null | When it last went live; null if never |

## `users/{uid}`

One per dashboard user, written by admins: a dashboard user (T303).

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `uid` | id |  | Their Firebase Auth uid |
| `email` | email |  | Their sign-in email |
| `role` | string | one of `admin`, `analyst`, `viewer` | Their role; T303 keeps it in step with the custom claim |
| `updated_at` | timestamp |  | When their role last changed |

## `outbox/{event_id}`

One per event whose changes Pub/Sub does not have yet, written by the API: an event's changes on their way to BigQuery (T306).

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `event_id` | UUID, as text |  | The event that wrote them, as text |
| `messages` | list of strings |  | One Pub/Sub message per document the event wrote: a row of raw.changes, as JSON text |
| `created_at` | timestamp |  | When the API wrote them: each message's written_at |
