## 🧪 Task 5: A/B Testing & Website Optimization Engine

Implemented statistical hypothesis testing framework to analyze homepage redesign variations using conversion rate Z-tests and session duration Welch t-tests.

```mermaid
graph TD
    A[GA4 / Web Analytics Log Data] --> B[A/B Variant Segmentation]
    B --> C[Z-Test for Conversion Lift]
    B --> D[Welch t-Test for Session Duration]
    C --> E[Statistically Significant Victory for Treatment B]
    D --> E
    E --> F[Interactive Streamlit Dashboard & Deployment]
```

### Key Statistical Results:
* **Conversion Rate Lift:** +25.7% relative increase (Control: 10.5% → Treatment: 13.2%, `p < 0.001`)
* **Bounce Rate Reduction:** -15.5% drop in bounce rate on Treatment variant.
* **Avg Session Duration:** Increased by 25 seconds per user.