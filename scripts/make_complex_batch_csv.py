"""Generate a 400-row complex bilingual batch CSV for app testing."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_batch_reviews_very_complex.csv"

PRODUCTS = [
    "phone",
    "shoes",
    "earbuds",
    "charger",
    "watch",
    "bag",
    "bottle",
    "lamp",
    "mouse",
    "keyboard",
    "cover",
    "power bank",
    "trimmer",
    "iron",
    "mixer",
    "fan",
    "heater",
    "router",
    "ssd",
    "camera",
]

TEMPLATES: list[tuple[str, str, str]] = [
    # english positive
    ("english", "clear_positive", "Excellent {p} quality and fast delivery."),
    ("english", "clear_positive", "I love this {p}, perfect fit and premium feel."),
    ("english", "clear_positive", "Great value {p}, highly recommended."),
    ("english", "clear_positive", "The {p} works exactly as described."),
    ("english", "clear_positive", "Very comfortable {p} for daily use."),
    ("english", "clear_positive", "Seller was helpful and the {p} packaging was secure."),
    ("english", "clear_positive", "Perfect {p} gift, everyone was delighted."),
    ("english", "clear_positive", "Smooth {p} performance with no lag."),
    ("english", "clear_positive", "Bright display and responsive {p} controls."),
    ("english", "clear_positive", "Worth buying this {p}, will order again."),
    # english negative
    ("english", "clear_negative", "Terrible {p} quality, broke after one day."),
    ("english", "clear_negative", "This {p} is not good, waste of money."),
    ("english", "clear_negative", "Late delivery and the {p} box was damaged."),
    ("english", "clear_negative", "{p} battery drains too fast, very disappointing."),
    ("english", "clear_negative", "Cheap {p} material and poor stitching."),
    ("english", "clear_negative", "Seller ignored my messages about the {p}."),
    ("english", "clear_negative", "Fake {p}, do not buy."),
    ("english", "clear_negative", "Returned the {p} because quality is bad."),
    ("english", "clear_negative", "Worst {p} I have ever used."),
    ("english", "clear_negative", "The {p} does not work as advertised."),
    # roman urdu positive
    ("roman_urdu", "clear_positive", "Yeh {p} bohat acha hai, mein khush hun."),
    ("roman_urdu", "clear_positive", "{p} quality zabardast hai packing bhi solid thi."),
    ("roman_urdu", "clear_positive", "Seller ne {p} jaldi deliver kiya shukriya."),
    ("roman_urdu", "clear_positive", "{p} battery timing kamaal hai charge jaldi hota hai."),
    ("roman_urdu", "clear_positive", "{p} size bilkul sahi aaya comfortable hai."),
    ("roman_urdu", "clear_positive", "Original {p} mila fragrance bhi lasting hai."),
    ("roman_urdu", "clear_positive", "{p} design stylish hai look premium lagta hai."),
    ("roman_urdu", "clear_positive", "Gift ke liye {p} perfect tha sab ko pasand aaya."),
    ("roman_urdu", "clear_positive", "{p} performance smooth hai hang nahi hota."),
    ("roman_urdu", "clear_positive", "Mein yeh {p} dobara kharidunga inshallah."),
    # roman urdu negative
    ("roman_urdu", "clear_negative", "Yeh {p} bilkul bekaar hai."),
    ("roman_urdu", "clear_negative", "{p} quality kharab hai ek din mein toot gaya."),
    ("roman_urdu", "clear_negative", "{p} delivery late thi box bhi damage tha."),
    ("roman_urdu", "clear_negative", "{p} battery bohat jaldi khatam ho jati hai."),
    ("roman_urdu", "clear_negative", "Seller ne {p} ke bare mein reply nahi diya."),
    ("roman_urdu", "clear_negative", "Fake {p} maal hai mat kharidna."),
    ("roman_urdu", "clear_negative", "{p} size chota aaya size chart ghalat hai."),
    ("roman_urdu", "clear_negative", "Mein is {p} purchase se khush nahi hun."),
    ("roman_urdu", "clear_negative", "Return karna para {p} quality bohat buri thi."),
    ("roman_urdu", "clear_negative", "Nahi yeh {p} lene ke qabil nahi."),
    # code mixed positive
    ("code_mixed", "clear_positive", "{p} quality bohat achi hai and delivery was fast."),
    ("code_mixed", "clear_positive", "Very happy with this {p}, zabardast experience."),
    ("code_mixed", "clear_positive", "{p} packing secure thi and item exact as photos."),
    ("code_mixed", "clear_positive", "Seller cooperative tha {p} issues jaldi resolve hue."),
    ("code_mixed", "clear_positive", "Worth buying hai this {p} will order again."),
    ("code_mixed", "clear_positive", "{p} performance smooth hai multitasking easy hai."),
    ("code_mixed", "clear_positive", "Genuine {p} mila and quality looks premium."),
    ("code_mixed", "clear_positive", "Overall experience positive tha with this {p}."),
    ("code_mixed", "clear_positive", "{p} size bilkul sahi aaya comfortable for daily use."),
    ("code_mixed", "clear_positive", "Arrived on time and {p} quality zabardast."),
    # code mixed negative
    ("code_mixed", "clear_negative", "{p} quality achi nahi hai and material feels cheap."),
    ("code_mixed", "clear_negative", "Not satisfied with {p}, bilkul waste of money."),
    ("code_mixed", "clear_negative", "Delivery late thi and {p} bhi damaged aaya."),
    ("code_mixed", "clear_negative", "Seller ne reply nahi diya about my {p} order."),
    ("code_mixed", "clear_negative", "Fake {p} hai do not buy please original nahi mila."),
    ("code_mixed", "clear_negative", "{p} battery drains fast hai very disappointing."),
    ("code_mixed", "clear_negative", "Size chart misleading hai {p} too small aaya."),
    ("code_mixed", "clear_negative", "Returned {p} because quality bohat kharab thi."),
    ("code_mixed", "clear_negative", "Worst {p} experience ever kabhi nahi lunga dobara."),
    ("code_mixed", "clear_negative", "Overpriced {p} hai and performance weak hai."),
    # linguistic probes
    ("english", "negation", "This {p} is not good at all."),
    ("english", "contrast", "I like this {p} design but battery life is terrible."),
    ("english", "contrast_pos", "Delivery was late but the {p} quality is excellent."),
    ("english", "sarcasm", "Oh wonderful, another broken {p}. Great service."),
    ("roman_urdu", "negation", "Yeh {p} acha nahi hai bilkul."),
    ("roman_urdu", "contrast", "{p} acha hai lekin quality kharab hai."),
    ("roman_urdu", "intensifier_pos", "Bohat zabardast {p} hai kamaal quality."),
    ("roman_urdu", "intensifier_neg", "Bohat bura {p} experience tha dobara nahi lunga."),
    ("code_mixed", "contrast", "{p} quality achi hai but delivery late thi."),
    ("code_mixed", "contrast", "Delivery late thi lekin {p} zabardast hai."),
    ("code_mixed", "negation", "I am not happy with this {p} bilkul bekaar nikla."),
    ("code_mixed", "url_noise", "Nice {p} check https://example.com/item{n} @seller #deal"),
    ("code_mixed", "punct_noise", "!!! WORST {p} ??? never again.... packing ZERO!!!!"),
    ("code_mixed", "informal_pos", "Qlty of {p} zabrdast hai price bhi ok hy."),
    ("code_mixed", "informal_neg", "Qlty of {p} kharab hai price bhi ziada hy."),
]


def main() -> None:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    n = 1

    for product in PRODUCTS:
        for lang, cat, tmpl in TEMPLATES:
            text = tmpl.format(p=product, n=n)
            n += 1
            text = " ".join(text.split())
            if text in seen:
                continue
            seen.add(text)
            rows.append(
                {
                    "review_id": len(rows) + 1,
                    "language": lang,
                    "category": cat,
                    "review": text,
                }
            )
            if len(rows) == 400:
                break
        if len(rows) == 400:
            break

    # Top up with numbered unique leftovers if needed
    k = 1
    while len(rows) < 400:
        candidates = [
            (
                "code_mixed",
                "extra",
                f"Extra test review {k}: product theek hai but delivery late thi case {k}.",
            ),
            (
                "english",
                "extra",
                f"Extra English review {k}: quality is good and shipping was fast case {k}.",
            ),
            (
                "roman_urdu",
                "extra",
                f"Extra Roman Urdu review {k}: quality kharab hai mat kharidna case {k}.",
            ),
        ]
        for lang, cat, text in candidates:
            if text in seen:
                continue
            seen.add(text)
            rows.append(
                {
                    "review_id": len(rows) + 1,
                    "language": lang,
                    "category": cat,
                    "review": text,
                }
            )
            if len(rows) == 400:
                break
        k += 1

    rows = rows[:400]
    for i, row in enumerate(rows, start=1):
        row["review_id"] = i

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["review_id", "language", "category", "review"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT}")
    print(f"unique texts: {len({r['review'] for r in rows})}")
    print("language counts:", dict(Counter(r["language"] for r in rows)))


if __name__ == "__main__":
    main()
