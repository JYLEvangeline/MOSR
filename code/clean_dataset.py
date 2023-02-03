import json
import pandas as pd
import re
import math
from selectors import EpollSelector
from utils import name_remove_blank,name_remove_non_alpha, filter_out 
# deal with the names in cc
path = '../data/'
dataset = ['soc-sign-bitcoinotc.csv',\
    'soc-redditHyperlinks-body.tsv',\
        'soc-redditHyperlinks-title.tsv',\
            'email-Eu-core-temporal.txt',\
                'CollegeMsg.txt',\
                'enron_cmu.csv',\
                'sorted_enron.csv',\
                    'emails.csv',\
                        'sorted_emails1.csv']
# read original CSV
csv_file = pd.read_csv(path + dataset[8])
# get name -> email dic
org = pd.read_csv(path + 'organazition2.csv')
dic_stage = {}
for i in range(len(org)):
    stage = org.iloc[i]['Stage'] 
name_email = org.loc[:,['Name','Email']]
name_email['Name'] = name_email['Name'].str.lower()
name_email = name_email.set_index('Name').T.to_dict('list')
# Clean CC
others = set()
content_type = set()
for i in range(0,len(csv_file)):
    content_type.add(csv_file.iloc[i]['Content-Type'])
    try:
        math.isnan(csv_file.iloc[i]['X-cc']) == False
        cc_receiver = []
    except:
        # print(csv_file.iloc[i]['X-cc'])
        a = re.findall(r"CN=(.*?)>", csv_file.iloc[i]['X-cc'])
        cc_receiver = [aa.split("CN=")[-1].lower()+"@enron.com" for aa in a]
        if a == []:
            a = re.findall(r"<(.*?)>", csv_file.iloc[i]['X-cc'])
            cc_receiver = [aa.lower() for aa in a]
            if a == []:
                print(csv_file.iloc[i]['X-cc'])
                a = re.findall(r"'(.*?)'", csv_file.iloc[i]['X-cc'])
                cc_receiver = [aa.lower() for aa in a]
                if a == []:
                    a = csv_file.iloc[i]['X-cc'].split(",")
                    cc_receiver = [aa.lower() for aa in a]
                    if a == []:
                        print(1)
            else:
                if "list not shown" in a[0]:
                    cc_receiver = []
    if cc_receiver != []:
        if '@' not in cc_receiver[0]:
            if len(cc_receiver) == 1:
                name = cc_receiver[0].split("/")[-1]
                if 'cn=' in name:
                    email = [name.split("cn=")[-1]+"@enron.com"]
                    cc_receiver = email
                else:
                    if name[0] not in name_email:
                        others.add(name)
                        tmp += 1
            else:
                tmp = 0
                prev_len = len(others)
                for i,name in enumerate(cc_receiver):
                    name = name_remove_blank(name)
                    cc_receiver[i] = name
                    if name not in name_email:
                        others.add(name)
                        tmp += 1
                new_len = len(others)

    csv_file.iloc[i]['X-cc'] = cc_receiver

csv_file.to_csv(path + 'cleaned_emails.csv', index = False)
csv_file1 = pd.read_csv(path + 'cleaned_emails.csv')
print(1)
print(content_type)