from flask import Blueprint,render_template,request,flash,url_for,app,redirect,session
import pandas as pd
import numpy as np
import string
from .static.crosswalk_symptom_inventories import set_crosswalk_files,crosswalk_scores
from collections import OrderedDict
from os import path
views = Blueprint('views',__name__)

script_dir = path.dirname(path.abspath(__file__))

@views.route('/',methods=['POST','GET'])
def home():
    input_rows = np.asarray([])
    output_rows = np.asarray([])
    inv_input = ""
    inv_output= ""
    if request.method == "POST":
        inv_input = request.form.get('input_select')
        inv_output = request.form.get('output_select')

        if inv_input == None or inv_output == None:
            flash('Please select the categories',category='error')
        elif inv_input == inv_output:
            flash('Please select different categories for input and output',category='error')
        else:
            input_rows,output_rows = get_table(inv_input,inv_output)
            session['inv_input'] = inv_input
            session['inv_output'] = inv_output
            session['input_rows'] = input_rows
            session['output_rows'] = output_rows
    return render_template("home.html",input_rows=input_rows,output_rows=output_rows,inv_input= inv_input,inv_output= inv_output)

def get_table(input,output):
    A = score_conversion(input,output)

    groups_path = script_dir + f"//static//groups.p"
    groups = pd.read_pickle(groups_path)
    groups = OrderedDict(sorted(groups.items()))
    input_titles ={}
    output_titles={}
    inp_count=1
    for key in groups.keys():
        if input in key:
            count_arr = [int(i) for i in range(inp_count,len(groups[key])+inp_count)]
            input_titles[key[1]] = list(zip(count_arr,groups[key]))
            inp_count+=len(groups[key])
        if output in key:
            # output_titles.append((key[1],groups[key]))
            output_titles[key[1]] = groups[key]
    # input_titles = A.text_dict[input]
    # output_titles = A.text_dict[output]
    # count_arr = [int(i) for i in range(1,len(input_titles)+1)]
    # return list(zip(count_arr,input_titles)),output_titles
    
    return input_titles,output_titles

@views.route("/convert",methods=["POST","GET"])
def convert():
    output_zip = []
    if request.method=="POST":
        scores = [request.form.get(i) for i in request.form]
        scores_ref = [i for i in request.form]
        print(scores)
        for i in range(len(scores)):
            session[scores_ref[i]] = scores[i]
        
        scores = list(map(int,scores))
        print("Session input: ",session['input_rows'])
        print("Session output: ",session['output_rows'])
        
        A = score_conversion(session['inv_input'],session['inv_output'])
        input_text_rows = A.text_dict[session['inv_input']]
        output_text_rows = A.text_dict[session['inv_output']]
        input_rows = []
        output_rows = []
        for key in session['input_rows'].keys():
            input_rows.extend(session['input_rows'][key])
        for key in session['output_rows'].keys():
            output_rows.extend(session['output_rows'][key])

        order = []
        for i in range(len(input_rows)):
            order.append(input_text_rows.index(input_rows[i][1]))
        reorder_scores = [x for _,x in sorted(zip(order,scores))]
        # print("Session output: ",session["inv_output"])
        predicted_scores = crosswalk_scores(
                            input_scores = reorder_scores,
                            score_dict = A.score_dict,
                            text_dict = A.text_dict,
                            hist_dict = A.hist_dict,
                            simil_arr = A.simil_arr,
                            empirical_shift_down = True,
                            inv_in = session['inv_input'],
                            inv_out = session['inv_output'],
                            verbose= False,
                            link_hists=True,
                            random_seed = 42,
                            )
        order= []
        for i in range(len(output_text_rows)):
            order.append(output_rows.index(output_text_rows[i]))
        
        # output_scores = []
        # for score in predicted_scores:
        #     if ~np.isnan(score):
        #         output_scores.append(int(round(score)))
        #     else:
        #         output_scores.append(float('nan'))
        print("Output rows: ",output_rows)
        print("Input rows: ",input_rows)
        print("Scores: ",scores)
        print("Reorder scores : ",reorder_scores)
        print("output text rows: ",output_text_rows)
        print("Order: ",order)
        predicted_scores = [int(i) for i in predicted_scores]
        final_scores = [x for _,x in sorted(zip(order,predicted_scores))]
        count = 0
        outdict = session['output_rows'].copy()
        for key in outdict.keys():
            outdict[key] = list(zip(final_scores[count:len(outdict[key])+count],outdict[key]))
            count+= len(outdict[key])
        print("Predicted scores before reordering: ",predicted_scores) 
        print("Output dict: ",outdict)
    return render_template('convert.html',
                            input_rows=session['input_rows'],
                            inv_input=session['inv_input'],
                            inv_output=session['inv_output'],
                            output_zip = outdict,
                            scores=scores,
                            input_score_sum=np.nansum(scores),
                            output_score_sum= int(np.sum(predicted_scores))
                            )

def score_conversion(inventory_in,inventory_out):
    score_dict_path = script_dir + f"//static//score_dict.p"
    text_dict_path = script_dir + f"//static//text_dict.p"
    hist_dict_path = script_dir + f"//static//hist_dict.p"
    A = set_crosswalk_files( 
                            score_file = score_dict_path,
                            text_file = text_dict_path,
                            hist_file = hist_dict_path,
                            inv_in = inventory_in,
                            inv_out = inventory_out,)
                            
    return A
