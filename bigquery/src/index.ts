/**
 * T306: every change to the collections the API writes, streamed to BigQuery.
 *
 * Firebase's function kit for Stream Firestore to BigQuery, the successor of
 * the extension. firebase.json lists one instance per collection; each one
 * deploys these functions as kit-<instance>-*, with its settings in
 * bigquery/instances/<instance>/.env, and writes the raw dataset's
 * <table>_raw_changelog table and <table>_raw_latest view.
 *
 * The functions run in Singapore because the kit retries failed inserts and
 * creates its tables through Cloud Tasks, which Bangkok (asia-southeast3)
 * doesn't offer. DATABASE_REGION is left blank so that this region applies.
 * The CLI still puts each Firestore trigger in the database's own region,
 * and the dataset is in Bangkok, so no data is stored outside Thailand.
 */
import { setGlobalOptions } from "firebase-functions";

setGlobalOptions({ region: "asia-southeast1" });

export * from "@firebase-function-kits/firestore-bigquery-export";
