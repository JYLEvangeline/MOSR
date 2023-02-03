import json
import pandas as pd
import re
from selectors import EpollSelector
from utils import filter_out, name_remove_blank, name_remove_non_alpha
# deal with the names in cc
path = '../data/'
file_name = "chat_distance2_cleaned.json"
f = open(path + file_name,'r')
chat = json.load(f)
org = pd.read_csv(path + 'organazition2.csv')
dic_stage = {}
for i in range(len(org)):
    dic_stage[org.iloc[i]['Email']] = org.iloc[i]['Stage'] 
name_email = org.loc[:,['Name','Email']]
name_email['Name'] = name_email['Name'].str.lower()
name_email = name_email.set_index('Name').T.to_dict('list')

def main2():
    # get set name
    dataset = ['soc-sign-bitcoinotc.csv',\
    'soc-redditHyperlinks-body.tsv',\
        'soc-redditHyperlinks-title.tsv',\
            'email-Eu-core-temporal.txt',\
                'CollegeMsg.txt',\
                'enron_cmu.csv',\
                'sorted_enron.csv',\
                    'emails.csv',\
                        'sorted_emails1.csv']
    csv_file = pd.read_csv(path + dataset[8])
    others = {'terrie smith', 'ronald brzezinski', 'mark fisher', 'ruth jensen', 'lois woodland', 'bret reich', 'russell leach', 'jeff duff', 'hollis kimbrough', 'jeff ghilardi', 'matt allsup', 'dave schulgen', 'martin essing', 'rick craig', 'william kendrick', 'olivia martinez', 'alan nueman', 'marion horna', 'jay godfrey', 'mark eilers', 'rec-scada', 'garth ripton', 'marc adler', 'donna martens', 'john nemila', 'kelly chambers', 'keith warner', 'ronda foster', 'leo nichols', 'kyle purvis', 'vwolf@swlaw.com', 'arnold l eisenstein', 'john shafer', 'mark ratekin', 'mark perryman', 'louis soldano', 'david miller', 'jimmy chandler', 'bryan gregory', 'byron rance', 'mike riedel', 'phil waddell', 'john lapham', 'richard melton', 'julie johnson', 'mark v walker', 'gary maestas', 'harvey kaura', 'earl chanley', 'alaadin suliman', 'patricia hunter', 'danielle lewis', 'bo thisted', 'leland meth', 'kevin cousineau', 'randy rice', 'joe thorpe', 'mike abbott', 'allan schoen', 'matthew meyers', 'flemming pedersen', 'benjamin bell', 'dan pribble', 'kim nguyen', 'rabi sahoo', 'rick loveless', 'jerry d martin', 'stoney buchanan', 'dan holli', 'robert grant', 'dan lindquist', 'james pfeffer', 'ron beidelman', 'clemens w"ste', 'jeff marecic', 'ilan caplan', 'guy dees', 'randy guire', 'sunitha kongara', 'david parham', 'jerry holt', 'dave sweet', 'tom nemila', 'rich jolly', 'jeff maurer', 'george griese', 'kurt anderson', 'emil moroz', 'michael miller', 'john m ruiz', 'val artman', 'james fleak', 'marilyn june', 'david roensch', 'melinda hood', 'jacob krautsch', 'clyde moter', 'joe chapman'}
    others_dic = {}
    start_i = 400000
    for i in range(start_i,len(csv_file)):
        # a = re.findall(r"\"(.*?)\"", csv_file.iloc[i]['X-From'])
        # if len(a) == 0:

        # email = csv_file.iloc[i]['From']
        # if len(a) > 1 or len(a) < 1:
        #     print(1)
        try:
            s = filter_out(csv_file.iloc[i]['X-From'])
            s = re.sub("[^\ a-zA-Z\d,\.]+",'',s)
            s = s.strip()
            if "," in s:
                s = s.split(",")
                new_s = s[1:] + [" "]+ [s[0]]
                name = "".join(new_s)
                name = name.strip()
            else:
                name = s
            if name in others:
                others_dic[name] = csv_file.iloc[i]['From']
        except:
            a = 1
    print(others_dic)
        
def main1():
    # get the set and clean cc
    others = set()
    new_dic = {}
    for e_i in chat:
        for i, content in enumerate(chat[e_i]):
            cc = content[-1]
            if cc != []:
                if '@' not in cc[0]:
                    if len(cc) == 1:
                        name = cc[0].split("/")[-1]
                        if 'cn=' in name:
                            email = [name.split("cn=")[-1]+"@enron.com"]
                            content[-1] = email
                        else:
                            if name[0] not in name_email:
                                others.add(name)
                                tmp += 1
                    else:
                        tmp = 0
                        prev_len = len(others)
                        for i,name in enumerate(cc):
                            name = name_remove_blank(name)
                            cc[i] = name
                            if name not in name_email:
                                others.add(name)
                                tmp += 1
                        new_len = len(others)
                        # if tmp == len(cc) and prev_len!= new_len:
                        #     print(1)
        new_e_i = name_remove_blank(e_i)
        new_dic[new_e_i] = chat[e_i]

    json_people = json.dumps(new_dic)
    f = open(path + "chat_distance2_cleaned.json","w")
    f.write(json_people)
    f.close()
main1()
