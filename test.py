from flask import Flask,render_template,request
import pandas as pd
import joblib
import xgboost as xgb


app=Flask(__name__)

@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        # clf = joblib.load("clf.pkl")
        
        # Get values through input bars
        customer_id = request.form["customer_id"]
        acc_length = request.form["acc_length"]
        loc_code=request.form["loc_code"]
        intl_plan=request.form["int_plan"]
        voice_plan=request.form["voice_plan"]
        num_vm=request.form["num_msg"]
        day_min=request.form["day_min"]
        day_call=request.form["day_call"]
        day_charge=request.form["day_charge"]
        eve_min=request.form["eve_min"]
        eve_call=request.form["eve_call"]
        eve_charge=request.form["eve_charge"]
        night_min=request.form["night_min"]
        night_call=request.form["night_call"]
        night_charge=request.form["night_charge"]
        intl_min=request.form["intl_min"]
        intl_call=request.form["intl_calls"]
        intl_charge=request.form["intl_charge"]
        cus_call=request.form["cus_call"]

        tot_charge=float(day_charge)+float(eve_charge)+float(night_charge)
        tot_min=float(day_min)+float(eve_min)+float(night_min)
        day=float(day_call)+float(day_charge)+float(day_min)

        
        # Put inputs to dataframe
        T = pd.DataFrame([[customer_id,acc_length,loc_code,intl_plan,voice_plan,num_vm,day_min,day_call,day_charge,eve_min,eve_call,eve_charge,night_min,night_call,night_charge,intl_min,intl_call,intl_charge,cus_call,tot_charge,tot_min,day]], columns = ["customer_id",
 "account_length",
 "location_code",
 "intertiol_plan",
 "voice_mail_plan",
 "number_vm_messages",
 "total_day_min",
 "total_day_calls",
 "total_day_charge",
 "total_eve_min",
 "total_eve_calls",
 "total_eve_charge",
 "total_night_minutes",
 "total_night_calls",
 "total_night_charge",
 "total_intl_minutes",
 "total_intl_calls",
 "total_intl_charge",
 "customer_service_calls",
 "total_charge",
 "total_minutes",
 "day"])

        

        clf = joblib.load("clf.pkl")
        T.to_csv('test2.csv',index=False)

        T=T[['customer_service_calls','intertiol_plan','voice_mail_plan','total_day_min','total_day_charge','location_code','total_charge','total_minutes','total_intl_calls','total_intl_charge','number_vm_messages','day']]

        T.to_csv('test3.csv',index=False)

        test_df=pd.read_csv("test3.csv")
        
    #     # Get prediction
        prediction = clf.predict(test_df)[0]
        
    else:
        prediction = ""
        
    return render_template("prediction.html", output = prediction)

if __name__ == '__main__':
    app.run(debug=True)