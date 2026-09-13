## Findings Summary

This dataset covers 320 cleaned retail orders placed between
January 2023 and December 2023,
totaling $516,824.98 in sales and $64,774.79 in profit,
an overall profit margin of 12.53%. The average order value across
all regions was $1,615.08.

Looking at regional performance, the **South** region had the highest
average sale value per order at $1,677.99, ahead of the other
three regions, though the gap between regions was relatively narrow — no
single region dominates the business.

Category-level profitability tells a more interesting story than raw sales
volume: **Office Supplies** had the highest profit margin at 14.86%,
while **Furniture** had the lowest at 11.26%, despite
neither being the top category by total sales. This suggests margin and
volume don't move together here, and a volume-only view of "top performers"
would be misleading.

The correlation analysis showed unit price is more strongly related to sale
value (correlation of 0.74) than quantity is, meaning big
orders are usually driven by higher-priced items rather than customers
buying more units per order. An IQR-based outlier check on profit flagged
a meaningful share of orders as statistical outliers, mostly high-margin
Electronics and Office Supplies orders, worth a closer look in a follow-up
analysis.

One data-quality limitation carried over from cleaning: 8
orders have no linked customer_id and were kept rather than dropped, so any
customer-segment analysis built on the merged dataset slightly undercounts
total order volume for those 8 orders.
