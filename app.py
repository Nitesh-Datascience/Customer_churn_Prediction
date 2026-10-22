import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    d=pd.read_csv("data/Customer Churn.csv")
    d["TotalCharges"]=pd.to_numeric(d["TotalCharges"],errors="coerce")
    d["Churn_Label"]=d["Churn"].map({"Yes":"Churned","No":"Retained"})
    d["Tenure Group"]=pd.cut(
        d["tenure"], [-1, 6, 12, 24, 48, 72, 100],
        labels=["0–6", "7–12", "13–24", "25–48", "49–72", "73+"]
    )
    return d

@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.joblib")

df=load_data(); model=load_model()

X=df.drop(columns=["Churn","Churn_Label","customerID","Tenure Group"],errors="ignore")
y=(df["Churn"].str.lower()=="yes").astype(int)
_,Xte,_,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
MET={
"Accuracy":accuracy_score(yte,pred),
"Precision":precision_score(yte,pred),
"Recall":recall_score(yte,pred),
"F1 Score":f1_score(yte,pred),
"ROC-AUC":roc_auc_score(yte,prob)
}

st.markdown("""
<style>
.stApp{background:#0b1020}.block-container{max-width:1450px;padding-top:1rem}
.hero{padding:25px 30px;border-radius:20px;background:linear-gradient(135deg,#151d36,#111827,#17132d);border:1px solid #28324d;margin-bottom:18px}
.hero h1{margin:0;font-size:clamp(26px,3vw,40px);font-weight:800;letter-spacing:-.6px}
.hero p{color:#aab4c8;margin:8px 0 0;font-size:15px}
.section{font-size:22px;font-weight:750;margin:22px 0 12px}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#121b2e,#101522);border:1px solid #293650;border-radius:16px;padding:15px}
div[data-testid="stMetricLabel"]{color:#aab4c8}
.insight{background:#111827;border:1px solid #26334c;border-left:4px solid #8b5cf6;padding:14px;border-radius:12px;margin:8px 0}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{height:46px;border-radius:10px;padding:0 16px;background:#121827}
.stTabs [aria-selected="true"]{background:#312e81}
div[data-testid="stPlotlyChart"]{border:1px solid #202b40;border-radius:14px;padding:8px}
div[data-testid="stDataFrame"]{border:1px solid #293650;border-radius:12px}

</style>
""",unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>📊 Customer Churn Intelligence</h1>
<p>Turn customer data into retention decisions · Explore patterns, estimate risk, and evaluate model quality</p></div>""",unsafe_allow_html=True)
with st.expander("How to use this dashboard", expanded=False):
    st.markdown(
        "1. Use the filters to focus on a customer segment.  "
        "2. Explore **Overview** and **Churn Analysis** for patterns.  "
        "3. Open **Predict Churn** to estimate risk for an individual profile.  "
        "4. Review **Model Performance** before interpreting predictions.  "
        "5. Use **Customer Data** to search the available records."
    )

with st.container(border=True):
    st.markdown("**Dashboard Filters**")
    a,b,c,d,e=st.columns(5)
    with a: genders=st.multiselect("Gender",sorted(df.gender.dropna().unique()),default=sorted(df.gender.dropna().unique()))
    with b: contracts=st.multiselect("Contract",sorted(df.Contract.dropna().unique()),default=sorted(df.Contract.dropna().unique()))
    with c: internet=st.multiselect("Internet Service",sorted(df.InternetService.dropna().unique()),default=sorted(df.InternetService.dropna().unique()))
    with d: payment=st.multiselect("Payment Method",sorted(df.PaymentMethod.dropna().unique()),default=sorted(df.PaymentMethod.dropna().unique()))
    with e: senior=st.multiselect("Senior Citizen",[0,1],default=[0,1])

f=df[df.gender.isin(genders)&df.Contract.isin(contracts)&df.InternetService.isin(internet)&df.PaymentMethod.isin(payment)&df.SeniorCitizen.isin(senior)].copy()
total=len(f); churned=int((f.Churn=="Yes").sum()); rate=churned/total if total else 0
st.markdown('<div class="section">Executive Overview</div>',unsafe_allow_html=True)
k1,k2,k3,k4,k5=st.columns(5)
k1.metric("Total Customers",f"{total:,}",border=True)
k2.metric("Churned Customers",f"{churned:,}",border=True)
k3.metric("Churn Rate",f"{rate:.1%}",border=True)
avg_monthly = f.MonthlyCharges.mean() if total else 0
avg_tenure = f.tenure.mean() if total else 0
k4.metric("Avg. Monthly Charges", f"${avg_monthly:,.2f}" if total else "—", border=True)
k5.metric("Avg. Tenure", f"{avg_tenure:.1f} mo" if total else "—", border=True)

if total == 0:
    st.warning("No customers match the current filters. Select at least one option in each filter to display analytics.")

t1,t2,t3,t4,t5=st.tabs(["📈 Overview","🔎 Churn Analysis","🎯 Predict Churn","🤖 Model Performance","📋 Customer Data"])
layout=dict(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=45,b=10))

with t1:
    c1,c2=st.columns(2)
    with c1:
        q=f.Churn_Label.value_counts().reset_index(); q.columns=["Status","Customers"]
        fig=px.pie(q,names="Status",values="Customers",hole=.62,title="Customer Retention Mix",color="Status",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c2:
        q=f.groupby("Contract").agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="Contract",y="Churn Rate",text=q["Churn Rate"].map(lambda x:f"{x:.1%}"),title="Churn Rate by Contract")
        fig.update_yaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        q=f.groupby("Tenure Group",observed=False).agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="Tenure Group",y="Churn Rate",text=q["Churn Rate"].map(lambda x:f"{x:.1%}"),title="Churn Rate by Tenure")
        fig.update_yaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c4:
        q=f.groupby("InternetService").agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="InternetService",y="Churn Rate",text=q["Churn Rate"].map(lambda x:f"{x:.1%}"),title="Churn Rate by Internet Service")
        fig.update_yaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section">Key Insights</div>',unsafe_allow_html=True)
    contract_rates = f.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()) if total else pd.Series(dtype=float)
    internet_rates = f.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean()) if total else pd.Series(dtype=float)
    if not contract_rates.empty:
        hc = contract_rates.idxmax()
        st.markdown(f'<div class="insight">📌 Highest churn rate by contract in this filtered view: <b>{hc}</b> ({contract_rates.max():.1%}).</div>', unsafe_allow_html=True)
    if not internet_rates.empty:
        hi = internet_rates.idxmax()
        st.markdown(f'<div class="insight">📌 Highest churn rate by internet service in this filtered view: <b>{hi}</b> ({internet_rates.max():.1%}).</div>', unsafe_allow_html=True)

with t2:
    c1,c2=st.columns(2)
    with c1:
        q=f.groupby(["SeniorCitizen","Churn_Label"]).size().reset_index(name="Customers")
        q["SeniorCitizen"]=q["SeniorCitizen"].map({0:"Non-Senior",1:"Senior"})
        fig=px.bar(q,x="SeniorCitizen",y="Customers",color="Churn_Label",barmode="group",title="Senior Citizen vs Churn",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c2:
        q=f.groupby("PaymentMethod").agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="Churn Rate",y="PaymentMethod",orientation="h",title="Churn Rate by Payment Method",
            text=q["Churn Rate"].map(lambda x:f"{x:.1%}"))
        fig.update_xaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        fig=px.box(f,x="Churn_Label",y="MonthlyCharges",color="Churn_Label",title="Monthly Charges vs Churn",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c4:
        s=f.sample(min(2500,len(f)),random_state=42)
        fig=px.scatter(s,x="tenure",y="TotalCharges",color="Churn_Label",size="MonthlyCharges",
            hover_data=["Contract","InternetService"],title="Tenure vs Total Charges",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)

with t3:
    left,right=st.columns([1.25,.75])
    with left:
        with st.container(border=True):
            p1,p2,p3=st.columns(3)
            gender=p1.selectbox("Gender",["Female","Male"]); senior_c=p2.selectbox("Senior Citizen",[0,1]); tenure=p3.number_input("Tenure (months)",0,100,12)
            p1,p2,p3=st.columns(3)
            partner=p1.selectbox("Partner",["Yes","No"]); dependents=p2.selectbox("Dependents",["Yes","No"]); phone=p3.selectbox("Phone Service",["Yes","No"])
            p1,p2,p3=st.columns(3)
            internet_c=p1.selectbox("Internet Service",["DSL","Fiber optic","No"]); contract_c=p2.selectbox("Contract",["Month-to-month","One year","Two year"])
            payment_c=p3.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            p1,p2,p3=st.columns(3)
            monthly=p1.number_input("Monthly Charges",0.0,500.0,70.0); total_c=p2.number_input("Total Charges",0.0,10000.0,1000.0); paperless=p3.selectbox("Paperless Billing",["Yes","No"])
            p1,p2,p3=st.columns(3)
            multiple=p1.selectbox("Multiple Lines",["No","Yes","No phone service"]); security=p2.selectbox("Online Security",["Yes","No","No internet service"]); backup=p3.selectbox("Online Backup",["Yes","No","No internet service"])
            p1,p2,p3=st.columns(3)
            device=p1.selectbox("Device Protection",["Yes","No","No internet service"]); support=p2.selectbox("Tech Support",["Yes","No","No internet service"]); tv=p3.selectbox("Streaming TV",["Yes","No","No internet service"])
            movies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])
            run=st.button("🚀 Predict Customer Risk",type="primary",use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("### Risk Result")
            if run:
                row=pd.DataFrame([{"gender":gender,"SeniorCitizen":senior_c,"Partner":partner,"Dependents":dependents,"tenure":tenure,
                    "PhoneService":phone,"MultipleLines":multiple,"InternetService":internet_c,"OnlineSecurity":security,
                    "OnlineBackup":backup,"DeviceProtection":device,"TechSupport":support,"StreamingTV":tv,"StreamingMovies":movies,
                    "Contract":contract_c,"PaperlessBilling":paperless,"PaymentMethod":payment_c,"MonthlyCharges":monthly,"TotalCharges":total_c}])
                p=float(model.predict_proba(row)[0,1])
                if p<.30: risk="LOW RISK"; action="Maintain normal engagement."
                elif p<.60: risk="MEDIUM RISK"; action="Consider targeted engagement or service review."
                else: risk="HIGH RISK"; action="Prioritize retention outreach."
                st.metric("Churn Probability",f"{p:.1%}")
                st.progress(p)
                if p<.30: st.success(risk)
                elif p<.60: st.warning(risk)
                else: st.error(risk)
                st.write("**Suggested action:**",action)
            else: st.info("Enter customer details and click Predict Customer Risk.")

with t4:
    m1,m2,m3,m4,m5=st.columns(5)
    for col,(name,val) in zip([m1,m2,m3,m4,m5],MET.items()): col.metric(name,f"{val:.1%}",border=True)
    c1,c2=st.columns(2)
    with c1:
        cm=confusion_matrix(yte,pred)
        fig=px.imshow(cm,text_auto=True,x=["Predicted Stay","Predicted Churn"],y=["Actual Stay","Actual Churn"],title="Confusion Matrix",color_continuous_scale="Blues")
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fpr,tpr,_=roc_curve(yte,prob)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"Logistic Regression (AUC={MET['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",**layout)
        st.plotly_chart(fig,use_container_width=True)

with t5:
    search=st.text_input("Search Customer ID",placeholder="Type a customer ID...")
    view=f.copy()
    if search: view=view[view.customerID.astype(str).str.contains(search,case=False,na=False)]
    cols=["customerID","gender","SeniorCitizen","tenure","Contract","InternetService","PaymentMethod","MonthlyCharges","TotalCharges","Churn_Label"]
    st.dataframe(view[cols].sort_values("MonthlyCharges",ascending=False),use_container_width=True,hide_index=True)
    st.download_button(
        "⬇️ Download filtered customer data",
        data=view[cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_churn.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.caption("Customer Churn Intelligence · Python · Pandas · Scikit-learn · Plotly · Streamlit")
st.caption("For learning and decision support only. Predictions are estimates, not guarantees; validate model quality before operational use.")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    d=pd.read_csv("data/Customer Churn.csv")
    d["TotalCharges"]=pd.to_numeric(d["TotalCharges"],errors="coerce")
    d["Churn_Label"]=d["Churn"].map({"Yes":"Churned","No":"Retained"})
    d["Tenure Group"]=pd.cut(d["tenure"],[-1,6,12,24,48,72,100],
        labels=["0–6","7–12","13–24","25–48","49–72","73+"])
    return d

@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.joblib")

df=load_data(); model=load_model()

X=df.drop(columns=["Churn","Churn_Label","customerID","Tenure Group"],errors="ignore")
y=(df["Churn"].str.lower()=="yes").astype(int)
_,Xte,_,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
MET={
"Accuracy":accuracy_score(yte,pred),
"Precision":precision_score(yte,pred),
"Recall":recall_score(yte,pred),
"F1 Score":f1_score(yte,pred),
"ROC-AUC":roc_auc_score(yte,prob)
}

st.markdown("""
<style>
.stApp{background:#0b1020}.block-container{max-width:1450px;padding-top:1rem}
.hero{padding:25px 30px;border-radius:20px;background:linear-gradient(135deg,#151d36,#111827,#17132d);border:1px solid #28324d;margin-bottom:18px}
.hero h1{margin:0;font-size:38px;font-weight:800}.hero p{color:#9ca3af;margin:7px 0 0}
.section{font-size:21px;font-weight:750;margin:20px 0 10px}
div[data-testid="stMetric"]{background:#111827;border:1px solid #29334c;border-radius:15px;padding:14px}
.insight{background:#111827;border-left:4px solid #8b5cf6;padding:14px;border-radius:10px;margin:7px 0}
</style>
""",unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>📊 Customer Churn Intelligence</h1>
<p>Executive overview • Churn analytics • Individual risk prediction • Model performance</p></div>""",unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("**Dashboard Filters**")
    a,b,c,d,e=st.columns(5)
    with a: genders=st.multiselect("Gender",sorted(df.gender.dropna().unique()),default=sorted(df.gender.dropna().unique()))
    with b: contracts=st.multiselect("Contract",sorted(df.Contract.dropna().unique()),default=sorted(df.Contract.dropna().unique()))
    with c: internet=st.multiselect("Internet Service",sorted(df.InternetService.dropna().unique()),default=sorted(df.InternetService.dropna().unique()))
    with d: payment=st.multiselect("Payment Method",sorted(df.PaymentMethod.dropna().unique()),default=sorted(df.PaymentMethod.dropna().unique()))
    with e: senior=st.multiselect("Senior Citizen",[0,1],default=[0,1])

f=df[df.gender.isin(genders)&df.Contract.isin(contracts)&df.InternetService.isin(internet)&df.PaymentMethod.isin(payment)&df.SeniorCitizen.isin(senior)].copy()
total=len(f); churned=int((f.Churn=="Yes").sum()); rate=churned/total if total else 0
st.markdown('<div class="section">Executive Overview</div>',unsafe_allow_html=True)
k1,k2,k3,k4,k5=st.columns(5)
k1.metric("Total Customers",f"{total:,}",border=True)
k2.metric("Churned Customers",f"{churned:,}",border=True)
k3.metric("Churn Rate",f"{rate:.1%}",border=True)
k4.metric("Avg. Monthly Charges",f"${f.MonthlyCharges.mean():,.0f}",border=True)
k5.metric("Avg. Tenure",f"{f.tenure.mean():.1f} mo",border=True)

t1,t2,t3,t4,t5=st.tabs(["📈 Overview","🔎 Churn Analysis","🎯 Predict Churn","🤖 Model Performance","📋 Customer Data"])
layout=dict(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=45,b=10))

with t1:
    c1,c2=st.columns(2)
    with c1:
        q=f.Churn_Label.value_counts().reset_index(); q.columns=["Status","Customers"]
        fig=px.pie(q,names="Status",values="Customers",hole=.62,title="Customer Retention Mix",color="Status",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c2:
        q=f.groupby("Contract").agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="Contract",y="Churn Rate",text=q["Churn Rate"].map(lambda x:f"{x:.1%}"),title="Churn Rate by Contract")
        fig.update_yaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        q=f.groupby("Tenure Group",observed=False).agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="Tenure Group",y="Churn Rate",text=q["Churn Rate"].map(lambda x:f"{x:.1%}"),title="Churn Rate by Tenure")
        fig.update_yaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c4:
        q=f.groupby("InternetService").agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="InternetService",y="Churn Rate",text=q["Churn Rate"].map(lambda x:f"{x:.1%}"),title="Churn Rate by Internet Service")
        fig.update_yaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section">Key Insights</div>',unsafe_allow_html=True)
    hc=f.groupby("Contract").apply(lambda x:(x.Churn=="Yes").mean(),include_groups=False).idxmax()
    hi=f.groupby("InternetService").apply(lambda x:(x.Churn=="Yes").mean(),include_groups=False).idxmax()
    st.markdown(f'<div class="insight">📌 Highest contract churn segment in the current filter: <b>{hc}</b>.</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="insight">📌 Highest internet-service churn segment in the current filter: <b>{hi}</b>.</div>',unsafe_allow_html=True)

with t2:
    c1,c2=st.columns(2)
    with c1:
        q=f.groupby(["SeniorCitizen","Churn_Label"]).size().reset_index(name="Customers")
        q["SeniorCitizen"]=q["SeniorCitizen"].map({0:"Non-Senior",1:"Senior"})
        fig=px.bar(q,x="SeniorCitizen",y="Customers",color="Churn_Label",barmode="group",title="Senior Citizen vs Churn",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c2:
        q=f.groupby("PaymentMethod").agg(Customers=("Churn","size"),Churned=("Churn",lambda x:(x=="Yes").sum())).reset_index()
        q["Churn Rate"]=q.Churned/q.Customers
        fig=px.bar(q,x="Churn Rate",y="PaymentMethod",orientation="h",title="Churn Rate by Payment Method",
            text=q["Churn Rate"].map(lambda x:f"{x:.1%}"))
        fig.update_xaxes(tickformat=".0%"); fig.update_traces(textposition="outside"); fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        fig=px.box(f,x="Churn_Label",y="MonthlyCharges",color="Churn_Label",title="Monthly Charges vs Churn",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c4:
        s=f.sample(min(2500,len(f)),random_state=42)
        fig=px.scatter(s,x="tenure",y="TotalCharges",color="Churn_Label",size="MonthlyCharges",
            hover_data=["Contract","InternetService"],title="Tenure vs Total Charges",
            color_discrete_map={"Retained":"#34d399","Churned":"#fb7185"})
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)

with t3:
    left,right=st.columns([1.25,.75])
    with left:
        with st.container(border=True):
            p1,p2,p3=st.columns(3)
            gender=p1.selectbox("Gender",["Female","Male"]); senior_c=p2.selectbox("Senior Citizen",[0,1]); tenure=p3.number_input("Tenure (months)",0,100,12)
            p1,p2,p3=st.columns(3)
            partner=p1.selectbox("Partner",["Yes","No"]); dependents=p2.selectbox("Dependents",["Yes","No"]); phone=p3.selectbox("Phone Service",["Yes","No"])
            p1,p2,p3=st.columns(3)
            internet_c=p1.selectbox("Internet Service",["DSL","Fiber optic","No"]); contract_c=p2.selectbox("Contract",["Month-to-month","One year","Two year"])
            payment_c=p3.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            p1,p2,p3=st.columns(3)
            monthly=p1.number_input("Monthly Charges",0.0,500.0,70.0); total_c=p2.number_input("Total Charges",0.0,10000.0,1000.0); paperless=p3.selectbox("Paperless Billing",["Yes","No"])
            p1,p2,p3=st.columns(3)
            multiple=p1.selectbox("Multiple Lines",["No","Yes","No phone service"]); security=p2.selectbox("Online Security",["Yes","No","No internet service"]); backup=p3.selectbox("Online Backup",["Yes","No","No internet service"])
            p1,p2,p3=st.columns(3)
            device=p1.selectbox("Device Protection",["Yes","No","No internet service"]); support=p2.selectbox("Tech Support",["Yes","No","No internet service"]); tv=p3.selectbox("Streaming TV",["Yes","No","No internet service"])
            movies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])
            run=st.button("🚀 Predict Customer Risk",type="primary",use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("### Risk Result")
            if run:
                row=pd.DataFrame([{"gender":gender,"SeniorCitizen":senior_c,"Partner":partner,"Dependents":dependents,"tenure":tenure,
                    "PhoneService":phone,"MultipleLines":multiple,"InternetService":internet_c,"OnlineSecurity":security,
                    "OnlineBackup":backup,"DeviceProtection":device,"TechSupport":support,"StreamingTV":tv,"StreamingMovies":movies,
                    "Contract":contract_c,"PaperlessBilling":paperless,"PaymentMethod":payment_c,"MonthlyCharges":monthly,"TotalCharges":total_c}])
                p=float(model.predict_proba(row)[0,1])
                if p<.30: risk="LOW RISK"; action="Maintain normal engagement."
                elif p<.60: risk="MEDIUM RISK"; action="Consider targeted engagement or service review."
                else: risk="HIGH RISK"; action="Prioritize retention outreach."
                st.metric("Churn Probability",f"{p:.1%}")
                st.progress(p)
                if p<.30: st.success(risk)
                elif p<.60: st.warning(risk)
                else: st.error(risk)
                st.write("**Suggested action:**",action)
            else: st.info("Enter customer details and click Predict Customer Risk.")

with t4:
    m1,m2,m3,m4,m5=st.columns(5)
    for col,(name,val) in zip([m1,m2,m3,m4,m5],MET.items()): col.metric(name,f"{val:.1%}",border=True)
    c1,c2=st.columns(2)
    with c1:
        cm=confusion_matrix(yte,pred)
        fig=px.imshow(cm,text_auto=True,x=["Predicted Stay","Predicted Churn"],y=["Actual Stay","Actual Churn"],title="Confusion Matrix",color_continuous_scale="Blues")
        fig.update_layout(**layout); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fpr,tpr,_=roc_curve(yte,prob)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"Logistic Regression (AUC={MET['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",**layout)
        st.plotly_chart(fig,use_container_width=True)

with t5:
    search=st.text_input("Search Customer ID",placeholder="Type a customer ID...")
    view=f.copy()
    if search: view=view[view.customerID.astype(str).str.contains(search,case=False,na=False)]
    cols=["customerID","gender","SeniorCitizen","tenure","Contract","InternetService","PaymentMethod","MonthlyCharges","TotalCharges","Churn_Label"]
    st.dataframe(view[cols].sort_values("MonthlyCharges",ascending=False),use_container_width=True,hide_index=True)

st.markdown("---")
st.caption("Customer Churn Intelligence • Python • Pandas • Scikit-learn • Plotly • Streamlit")
