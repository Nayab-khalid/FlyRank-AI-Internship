Tell the Story

5 Minute Demo Outline

Question

The content team has many pages to review but limited time. My project asks whether observable search, content and freshness signals can help rank pages so reviewers can focus on pages associated with an observed declining trend.

Method

I framed the problem as binary classification.

I first created a transparent baseline using content staleness and search visibility.

I then developed a Logistic Regression model using measurable search, content, engagement and keyword context signals.

I used client grouped validation so that the same client did not appear in both the training and test groups.

I also performed a leakage check and removed features that were derived from the label.

Chart

I will show the model versus baseline comparison from the project.

The chart compares Average Precision, ROC AUC and Precision at 50.

Honest Result

On the held out client grouped test set, the corrected Logistic Regression model achieved an Average Precision of 0.596008, ROC AUC of 0.594877 and Precision at 50 of 0.70.

The baseline achieved an Average Precision of 0.489069, ROC AUC of 0.506511 and Precision at 50 of 0.30.

The test set positive rate was 51.1 percent.

The corrected model performed better than the baseline, but the improvement was modest. These are measured results on the evaluated test set and do not guarantee the same performance on future data.

Recommendation

The model should be used as a prioritization tool for human review.

Reviewers should start with the highest ranked pages, check the actual page and its search context, and then decide whether the page should be refreshed, protected and refreshed, reviewed for CTR, reviewed for engagement, expanded or monitored.

The model should not automatically change, delete or publish content.


Social Post

I built a machine learning workflow for content refresh prioritization as part of the FlyRank ML Internship.

The project asks a practical question: when a content team has many pages to review, can measurable search, content and freshness signals help decide where to look first?

I compared a transparent baseline with Logistic Regression, used client grouped validation and performed leakage checks to make the evaluation more honest.

On the held out test set, the corrected model achieved 0.5960 Average Precision compared with 0.4891 for the baseline.

The biggest lesson was not simply getting a higher score. It was learning to check the signals, identify leakage, use an honest validation design and keep the final output as decision support rather than treating the model as an automatic decision maker.


Employer Facing Summary

I built a machine learning based content prioritization workflow using the anonymized FlyRank content dataset to help a content team decide which pages should be reviewed first.

I created a transparent baseline, developed a Logistic Regression model, performed leakage checks and evaluated the model using a client grouped validation split.

On the held out test set, the corrected model achieved 0.5960 Average Precision compared with 0.4891 for the baseline, showing measured improvement in ranking pages associated with the observed decline label.
