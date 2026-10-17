# Kaizen Labs

Turns a multi-outlet F&B operator's POS data into next week's plan: customers per day, people on
each shift, what each menu item will sell, the ingredients that takes, and an order list. Shown on
one dashboard page and a one-page PDF. Also holds the one-page website
for kaizenlabs.sg (`site/`).

```bash
make install                                # once
make sample && make forecast && make plan && make dashboard
```

Then open http://localhost:8501. `make report` writes a one-page PDF. `make test` runs the tests.

Everything runs on your laptop. No paid services, no accounts. The only internet calls are to
official Singapore public data (data.gov.sg and the Meteorological Service), and the results are
cached in `data/signals/` so the pipeline also runs offline.

---

## The steps, in plain English

Every step takes `CLIENT=<key>` (default `demo`, the sample café group). A client is a folder in
`clients/<key>/` with a `client.yaml` (where its files live, planning assumptions) and, for real
clients, a `mapping.yaml` (how to read their POS export).

| Command | What it does | Reads | Writes |
|---|---|---|---|
| `make sample` | Makes up 52 weeks of history plus next week for "Demo Café Group", five fictional outlets. Also writes two messy POS exports to practise `make ingest` on. | `data/signals/` | `data/sample/` |
| `make ingest CLIENT=x` | Turns a client's raw POS export into the standard input files, using their `mapping.yaml`, and checks them. | `data/clients/x/raw/` | `data/clients/x/inputs/` |
| `make validate CLIENT=x` | Checks a client's input files against the data contract and says what is wrong, line by line. | inputs | (nothing) |
| `make signals CLIENT=x` | Gets public holidays, school holidays and daily weather for the client's dates. **The only step that uses the internet.** `OFFLINE=1` uses the cache only. | inputs, `data/signals/` | `signals_daily.csv` |
| `make forecast CLIENT=x` | Runs `signals`, then tests the model on the last 8 weeks against the rules operators use today, prints the accuracy per outlet, and forecasts the next 7 days hour by hour. | inputs, `signals_daily.csv` | `forecast.csv`, `backtest*.csv`, `daily_sales.csv` |
| `make plan CLIENT=x` | Turns the forecast into people per shift, menu items per day, ingredients per day and an order list; compares the roster with what was needed over the last 28 days; makes a clearly labelled savings estimate. | inputs, `forecast.csv` | `staffing_plan.csv`, `sku_forecast.csv`, `ingredient_forecast.csv`, `suggested_orders.csv`, `outlet_summary.csv`, and supporting files |
| `make dashboard CLIENT=x` | Opens the dashboard. It reads only the files in the outputs folder, and each section names its source file. | outputs | (nothing) |
| `make report CLIENT=x` | One-page PDF summary from the same files. | outputs | `<key>_summary.pdf` |

Outputs go to `data/outputs/` for the demo and `data/clients/<key>/outputs/` for everyone else.

### 1. Sample data (`kaizen/sample.py`)

Daily transactions per outlet = base level × weekday pattern × public holiday × school holiday ×
season × rain × slow growth × noise, then spread over opening hours with each outlet's hourly shape.

| Outlet | Pattern built in |
|---|---|
| CBD - Raffles Place | Busy Mon-Thu, softer Fridays, near-empty weekends and public holidays, quieter on days next to a holiday (people take leave), morning and lunch spikes |
| Orchard | Weekend peaks, festive season, loses up to ~28% of trade on very wet days, afternoon and evening peaks |
| Tampines, Jurong East | Weekends and school holidays up, small rain effect |
| Changi Airport T3 | Steady, 6am to midnight, follows travel seasons (June, December), no rain effect |

Weather is **real** (MSS Changi daily records) where published, and synthetic for the rest, so a rainy
day in the demo really was rainy.

The menu has 8 items (flat white, latte, long black, iced latte, matcha latte, butter croissant,
chicken sandwich, banana bread), each with its own share of sales per outlet and a weekend effect.
`recipes.csv` breaks them into 12 ingredients (e.g. a flat white = 18 g beans, 150 ml milk, and a cup
and lid 60% of the time for takeaway). Stock, waste and rosters are simulated with the rules of
thumb operators use today: fixed par levels for ingredients, fresh items made at "last week plus
15%", and one roster template per weekday of three 6-hour shifts with 3 to 6 people, which ignores
holidays and seasons. That is where the demo's waste, stock-outs and roster gaps come from.

History ends last Sunday by default; `make sample END=2026-10-11` fixes the date. Same seed, same data.

### 2. The data contract (`kaizen/contract.py`)

What a client provides. Only `sales.csv` is required.

| File | Columns | Notes |
|---|---|---|
| `sales.csv` | `timestamp, outlet, transactions` | Hourly, start of the hour, Singapore time, no offset. Missing hours = no sales. |
| `items.csv` | `date, outlet, item, units_sold` | Optional. Menu items (SKUs) sold per day. Gives each outlet's sales mix. |
| `stock.csv` | `date, outlet, item, unit, on_hand` | Optional. Closing count per ingredient. Fresh items written off daily don't need counting. |
| `roster.csv` | `date, outlet, hour, scheduled_staff` | Optional. Include next week to compare against the suggested staffing. |
| `waste.csv` | `date, outlet, waste_sgd` | Optional. |
| `menu.csv` | `item, unit, usage_per_transaction` | Optional, agreed with the client. Menu items per transaction, used when there is no `items.csv`. |
| `recipes.csv` | `item, ingredient, unit, qty_per_item` | Optional, agreed with the client. The bill of materials: what one of each menu item uses. Without it, each menu item is its own stock item. |
| `weather.csv` | `date, rain_mm, max_temp_c` | Optional fallback weather. |

Files must have **exactly** these columns. A stray `customer_name` or `phone` column is rejected, not
carried along. Checks are written as pydantic models, one per file, plus duplicate and cross-file
checks (every outlet must appear in `sales.csv`). Errors point to the CSV line.

### 3. Ingest: onboarding a real client (`kaizen/ingest.py`)

1. Create `clients/<key>/client.yaml` (copy `clients/example_15min/client.yaml`; a name is enough to start).
2. Put their export(s) in `data/clients/<key>/raw/`. This folder is git-ignored: **client data is never committed**.
3. Write `clients/<key>/mapping.yaml`: which column holds the time, the outlet and the count, and
   the outlet names. Two worked examples:
   - `clients/example_15min/`: 15-minute buckets with `+08:00` timestamps, plus stock, waste, roster, menu and bill-of-materials files with their own column names and date formats.
   - `clients/example_txn/`: one row per receipt line, `dd/mm/yyyy` dates, voids mixed in. Receipts are counted once each, voids dropped.
4. `make ingest CLIENT=<key>`, then `make forecast plan dashboard CLIENT=<key>`.

Ingest reads **only** the columns named in the mapping, so customer details in an export are never
loaded. It also stops if the export has an outlet the mapping doesn't know (so nothing is silently
dropped), converts timestamps with offsets to Singapore time, and drops a partial last day.

Try it: `make sample` creates the raw files, then `make ingest CLIENT=example_15min` and
`make ingest CLIENT=example_txn`. Both reproduce the sample's hourly counts exactly.

### 4. Outside signals (`kaizen/signals.py`)

| Signal | Source | Cache |
|---|---|---|
| Public holidays | data.gov.sg "Singapore Public Holidays" (Ministry of Manpower), all years in the collection, including "observed" Mondays | `data/signals/public_holidays.csv` |
| School holidays | MOE term breaks, hand-maintained. **Add next year's dates when MOE publishes them (usually July)**; the pipeline warns if a year is missing. | `data/signals/school_holidays.yaml` |
| Weather | MSS historical daily records, station S24 (Changi): daily rainfall and max temperature | `data/signals/weather_mss_S24.csv` |

Each source has its own function and cached file. MSS publishes with a lag of a few weeks, and the
forecast week is in the future, so days without a record use the client's `weather.csv` if there is
one, otherwise the monthly average of recorded days. The log says how many days came from each
source, and every row of `signals_daily.csv` has a `weather_source` column.

### 5. Forecast (`kaizen/forecast.py`, `kaizen/features.py`)

- **Baselines** (what operators do today): "same hour last week" and "average of the same hour on the
  same weekday over the last 4 weeks".
- **Model**: one LightGBM model across all outlets (outlet is a category), Poisson objective. Features:
  hour, weekday, outlet, public holiday, school holiday, days to and from the nearest public holiday,
  rain, max temperature, sales at the same hour 7, 14 and 28 days ago, the 4-week same-hour average,
  the outlet's average daily total over the 7 days ending a week ago, and whether the day a week ago
  was a holiday (that is what trips up the last-week rule).
- **No peeking**: every feature uses only sales at least 7 days old, so one set of features serves all
  7 forecast days without feeding forecasts back in. A test checks this.
- **Back-test**: rolling origin over the last 8 weeks. Stand at the end of a week, train on everything
  before, forecast the next 7 days, move on a week. Accuracy = 1 − WAPE (total absolute error ÷ total
  actual transactions), per outlet, for the model and both baselines, at hourly and daily level.
- **Range**: two quantile models give an 80% range (`lower`, `upper`).

On the sample data (your numbers will vary slightly with the date):

```
7 days ahead, all outlets   model   last week   4-week avg
Daily totals                92.7%     87.9%       90.2%
Hourly                      79.8%     71.3%       77.5%
```

Hourly accuracy has a ceiling: at ~50 transactions an hour, pure chance alone moves an hour by
about 14%. The dashboard leads with daily accuracy, which is what ordering and the founder's
week depend on; shifts use the busiest hour, which the hourly forecast provides.

### 6. Plan (`kaizen/plan.py`)

- **People per shift.** Each day has three 6-hour shifts: Opening 06:00-12:00, Mid 12:00-18:00,
  Closing 18:00-24:00. People on a shift = enough for its **busiest hour**:
  ceil(busiest-hour forecast customers ÷ customers per person per hour), kept between 3 and 6. A
  shift with no customers expected is closed. If the busiest hour would need more than 6, the
  shift is rostered at 6 and flagged so the founder can add a part-timer for the peak. Shift times,
  the 3-to-6 range and the 30 customers per person per hour are all in `client.yaml`, and can differ
  per outlet. The plan also lists every shift where the current roster differs.
- **What will sell.** Forecast units of each menu item per day = forecast customers × that item's
  share of sales at that outlet over the last 28 days, worked out separately for weekdays and for
  weekends/public holidays (from `items.csv`; `menu.csv` if there's no history). Simple on purpose:
  the founder can check it against the till. The shares used are in `sku_mix.csv`.
- **Ingredients per day.** Menu items × `recipes.csv`.
- **Order list.** Ingredient use over its cover days (days until the next delivery) × (1 + buffer) −
  usable stock on hand, rounded up to whole packs, never negative. Fresh items (bread, lettuce,
  banana bread) cover tomorrow only and don't count leftovers. Buffer, pack size, cover days and
  perishable are per ingredient in `client.yaml` (default buffer 10%).
- **Gaps.** For each outlet, day and shift of the last 28 days: people rostered vs what the actual
  customers needed under the same rule, in staff-hours (`staffing_gaps.csv`).
- **Savings estimate.** Share of waste assumed avoidable + share of over-staffed hours assumed cut ×
  hourly wage, per month. All assumptions live in `client.yaml` and are written next to every number
  in `savings_estimate.csv`. It is always labelled an estimate, never a guarantee.

### 7. Dashboard and report

`dashboard/app.py` (Streamlit) is one page about **next week**, in the order a founder decides things:

1. **Headline numbers:** customers expected (vs last week), busiest day, people-shifts to roster
   (vs the current roster), number of ingredients to order. A short "needs a decision" list
   (public holidays, shifts that need more than 6).
2. **How busy each day will be:** 7 bars, with last week's figure as a tick.
3. **Who to roster:** people per shift per outlet for the 7 days, then the list of roster changes.
4. **What will sell:** each menu item per day.
5. **Ingredients and what to order:** daily use per ingredient, then the order list per outlet.

How the forecast was checked (accuracy vs "same as last week") and the last 28 days (waste, roster
gaps, stock-outs, savings estimate) sit in collapsed sections at the bottom. An outlet picker at the
top switches between the group and one outlet. Works on a phone. `make report` puts the same plan on
one A4 page.

Every number traces to a file:

| Dashboard section | File in the outputs folder |
|---|---|
| Customers per day | `daily_sales.csv` (from `forecast.csv`) |
| People per shift, roster changes | `staffing_plan.csv` |
| What will sell | `sku_forecast.csv` (shares used: `sku_mix.csv`) |
| Ingredients per day | `ingredient_forecast.csv` |
| Order list | `suggested_orders.csv` |
| How reliable | `backtest_accuracy.csv` (hourly detail in `backtest.csv`) |
| Last 28 days | `outlet_summary.csv` (detail: `staffing_gaps.csv`), `savings_estimate.csv` |

---

## Website (`site/`)

A single static `index.html`, no framework, no external requests (JetBrains Mono is self-hosted in
`site/assets/fonts/` under its open font licence), no forms, no trackers, no cookies. Light and dark
mode follow the visitor's system setting. Preview with `make site` (http://localhost:8000).

To refresh the dashboard screenshot: run `make dashboard`, then `make screenshot` (needs
`pip install playwright` and `playwright install chromium`).

### Deploying to GitHub Pages (free for this public repo)

1. Merge to `main`. `.github/workflows/pages.yml` publishes `site/` whenever it changes.
2. GitHub → repo **Settings → Pages** → Source: **GitHub Actions**.
3. Same page → Custom domain: `kaizenlabs.sg` (the `site/CNAME` file says the same). Recommended:
   verify the domain first under your GitHub **profile Settings → Pages → Add a domain**, which gives you a
   TXT record to add (below), so nobody else can claim it.
4. Add the DNS records below, wait for the check to pass, then tick **Enforce HTTPS**.

### DNS records for kaizenlabs.sg

Make these changes at whoever hosts your DNS (your .sg registrar, or Cloudflare if you moved it there).

**Add**

| Type | Name | Value |
|---|---|---|
| A | `@` | `185.199.108.153` |
| A | `@` | `185.199.109.153` |
| A | `@` | `185.199.110.153` |
| A | `@` | `185.199.111.153` |
| AAAA | `@` | `2606:50c0:8000::153` |
| AAAA | `@` | `2606:50c0:8001::153` |
| AAAA | `@` | `2606:50c0:8002::153` |
| AAAA | `@` | `2606:50c0:8003::153` |
| CNAME | `www` | `shauryabuilds.github.io.` |
| TXT | `_github-pages-challenge-shauryabuilds` | the code GitHub shows you in step 3 |

**Remove** only an existing `A`/`AAAA` record on `@` or a `www` record that points at a registrar
parking or "URL forwarding" page.

**Do not touch** your Google Workspace records. They sit alongside the ones above without conflict:

| Type | Name | Typical value |
|---|---|---|
| MX | `@` | `smtp.google.com` (priority 1), or on older setups `ASPMX.L.GOOGLE.COM` (1), `ALT1/ALT2.ASPMX.L.GOOGLE.COM` (5), `ALT3/ALT4.ASPMX.L.GOOGLE.COM` (10) |
| TXT | `@` | `v=spf1 include:_spf.google.com ~all` |
| TXT | `google._domainkey` | your DKIM key, `v=DKIM1; k=rsa; p=...` |
| TXT | `_dmarc` | e.g. `v=DMARC1; p=none; rua=mailto:shaurya@kaizenlabs.sg` |
| TXT | `@` | `google-site-verification=...` |

Two rules keep email safe: **never put a CNAME on `@`** (a CNAME can't sit next to MX and TXT
records, so it would break mail), and **add records rather than replace them**. Several TXT records
on `@` are normal. If your DNS host makes you edit `@` TXT as one combined entry, keep the SPF and
verification values and add nothing to them.

Check afterwards:

```bash
dig +short kaizenlabs.sg A          # the four 185.199.x.153 addresses
dig +short www.kaizenlabs.sg CNAME  # shauryabuilds.github.io.
dig +short kaizenlabs.sg MX         # still Google
```

If you prefer Cloudflare Pages: it's free, but the apex domain needs Cloudflare to run your DNS
(change nameservers at the registrar and re-create the Google records there first). GitHub Pages
avoids that move.

---

## Repo layout

```
kaizen/            pipeline: sample, contract, ingest, signals, features, forecast, plan, report, config
dashboard/app.py   Streamlit dashboard
clients/<key>/     client.yaml (+ mapping.yaml): demo, example_15min, example_txn
data/signals/      cached public data (committed) + school_holidays.yaml
data/sample/       generated demo inputs (git-ignored)
data/outputs/      demo outputs (git-ignored)
data/clients/      real client raw files, inputs, outputs (git-ignored)
site/              kaizenlabs.sg
tests/             data contract, ingest, features/back-test maths, orders and staffing
```

## Known limits

- Opening hours are inferred from the hours an outlet has traded; an outlet that opens new hours
  needs a few weeks before the model forecasts them well.
- People per shift is sized for the busiest hour, so quieter hours of a long shift carry slack; a
  split shift or part-timer is the fix when a shift is flagged.
- The menu-item forecast assumes next week's sales mix looks like the last 4 weeks' (by weekday vs
  weekend). A new menu item or a promotion needs its share set by hand in `menu.csv` at first.
- The forecast week uses average weather unless a forecast is supplied as `weather.csv`. Rain matters
  most for Orchard-type outlets.
- `school_holidays.yaml` must be updated once a year.
- Sales mix learned from `items.csv` slightly under-counts items on days they sold out.
