import pandas as pd

df = pd.read_csv("Dataset_DecisionPSS.csv",sep=";")

df.plot.scatter(x="ExamID",y="PSS_Stress");
df.plot.scatter(x="FinalGrade",y="PSS_Stress");
df.plot.scatter(x="TotalQuestions",y="PSS_Stress");
df.plot.scatter(x="avg_durationperquestion",y="PSS_Stress");
df.plot.scatter(x="decision_time_efficiency",y="PSS_Stress");
df.plot.scatter(x="avg_tbd",y="PSS_Stress");
df.plot.scatter(x="good_decision_time_efficiency",y="PSS_Stress");
df.plot.scatter(x="maxduration",y="PSS_Stress");
df.plot.scatter(x="median_tbd",y="PSS_Stress");
df.plot.scatter(x="minduration",y="PSS_Stress");
df.plot.scatter(x="num_decisions_made",y="PSS_Stress");
df.plot.scatter(x="question_enter_count",y="PSS_Stress");
df.plot.scatter(x="ratio_decisions",y="PSS_Stress");
df.plot.scatter(x="ratio_good_decisions",y="PSS_Stress");
df.plot.scatter(x="totalduration",y="PSS_Stress");
df.plot.scatter(x="variance_tbd",y="PSS_Stress");

df.hist(figsize = [30,25])

pd.plotting.scatter_matrix(df.loc[:,["PSS_Stress",
                                     "FinalGrade",
                                     "TotalQuestions",
                                     "avg_durationperquestion",
                                     "decision_time_efficiency",
                                     "avg_tbd",
                                     "good_decision_time_efficiency",
                                     "maxduration",
                                     "median_tbd",
                                     "minduration",
                                     "num_decisions_made",
                                     "question_enter_count",
                                     "ratio_decisions",
                                     "ratio_good_decisions",
                                     "totalduration",
                                     "variance_tbd",
                                     ]], figsize=(40,40),s=150,marker='D')