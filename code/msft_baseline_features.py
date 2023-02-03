from tkinter import W
import nltk
import time

from collections import defaultdict
from datetime import datetime, timedelta
from dateutil import parser
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from utils import dict_float, dict_int, dict_list, dict_time
from utils import get_mean_min_max, get_mean_med_time_list, time_to_hours, sentiment_words

# pre-defined
stopword=nltk.corpus.stopwords.words('english')
path = '../code/'
pos_words_set = sentiment_words(path + "positive-words.txt")
neg_words_set = sentiment_words(path + "negative-words.txt")


def address(email_item):
    """
    Group: Address
    IsInternalExternal
    NumOfRecipients
    """    
    # IsInternalExternal
    IsInternalExternal = [] # 1 as internal, 0 is external
    from_email = email_item['From']
    try:
        to_emails = email_item['To'].split(",")
    except:
        to_emails = []
    if 'enron' in from_email:
        for to_email in to_emails:
            internal_flag = 1 if 'enron' in to_email else 0
            IsInternalExternal.append(internal_flag)
    # NumOfRecipients
    NumOfRecipients = [len(to_emails)] * len(to_emails)
    
    return IsInternalExternal, NumOfRecipients

def BOW(email_item):
    print(1)

def CPred(email_item):
    """
    Group: CPred
    SentimentWords
    CommitmentScore
    RequestScore
    """
    content_without_stopwords = [w for w in nltk.word_tokenize(email_item['Content']) if w not in stopword]
    # SentimentWords
    pos_words_num =  sum([1 if w in pos_words_set else 0 for w in content_without_stopwords])
    neg_words_num =  sum([1 if w in neg_words_set else 0 for w in content_without_stopwords])
    SentimentWords = [pos_words_num, neg_words_num]
    # CommitmentScore
    # RequestScore
    # these two waiting to add. Need msft inside
    return SentimentWords

def CProp(email_item):
    """
    Group: CProp
    EmailSubLen
    EmailBodyLen
    """
    # EmailSubLen
    try:
        EmailSubLen = len(email_item['Subject'].split())
    except:
        EmailSubLen = 0
    # EmailBodyLen
    try:
        EmailBodyLen = len(email_item['Content'].split())
    except:
        EmailBodyLen = 0
    return EmailBodyLen, EmailSubLen

# This is a pre_process one
def pre_HistIndiv(emails):
    """
    Group: HistIndiv
    HistReplyRateGlobalUI
    HistReplyNumGlobalUI
    HistRecEmailNumSTGlobalUI
    HistRecEmailNumCCGlobalUI
    HistSentEmailNumGlobalUI
    HistReplyTimeMeanGlobalUI
    HistReplyTimeMedianGlobalUI
    HistGlobalUJ
    """
    # HistReplyRateGlobalUI
    HistReplyRateGlobalUI = defaultdict(float)
    
    # HistReplyNumGlobalUI
    HistReplyNumGlobalUI = defaultdict(int)
    
    # HistRecEmailNumSTGlobalUI
    HistRecEmailNumSTGlobalUI = defaultdict(int)
    
    # HistRecEmailNumCCGlobalUI
    HistRecEmailNumCCGlobalUI = defaultdict(int)
    
    # HistSentEmailNumGlobalUI
    HistSentEmailNumGlobalUI = defaultdict(int)
    
    # HistReplyTimeMeanGlobalUI
    HistReplyTimeMeanGlobalUI = defaultdict(float)
    
    # HistReplyTimeMedianGlobalUI
    HistReplyTimeMedianGlobalUI = defaultdict(float)
    
    # HistGlobalUJ
    # not available

    # history of previous
    # check "RE" in title
    waitinglist = defaultdict(dict_time) # history[w_i][w_j] = time -> w_j is waiting for the reply from w_i
    reply_time_list = defaultdict(list)
    for i in range(len(emails)):
        try:
            email = emails.iloc[i]
        except:
            email = emails[i]
        from_email = email['From']
        # to_email may be nan
        try:
            to_emails = email['To'].split()
        except:
            to_emails = []
        
        # cc email not applied in Enron
        try:
            cc_emails = email['CC'].split() # enron dosn't have cc, only X-cc
        except:
            cc_emails = []
        cur_time = parser.parse(email['Date'])
        
        for to_email in to_emails:
            waitinglist[to_email][from_email] = cur_time
            if to_email in waitinglist[from_email]:
                # From reply to To
                HistReplyNumGlobalUI['From'] += 1
                # preparation for ReplyTimeMean and Med
                # the average reply time of from_email (cur_time(reply time) - waitinglist[from_email][to_email](send time))
                reply_time_list[from_email].append(cur_time - waitinglist[from_email][to_email])
                # initialization
                del(waitinglist[from_email][to_email])
            HistRecEmailNumSTGlobalUI[to_email] += 1
            
        for cc_email in cc_emails:
            waitinglist[cc_email][from_email] = cur_time
            if cc_email in waitinglist[from_email]:
                # From reply to CC
                HistReplyNumGlobalUI['From'] += 1
                # preparation for ReplyTimeMean and Med
                # cc_email (second key) sends msg to from_email (first key) at waitinglist[from_email][cc_email]
                reply_time_list[from_email].append(cur_time - waitinglist[from_email][cc_email])
                # initialization
                del(waitinglist[from_email][cc_email])
            HistRecEmailNumCCGlobalUI[cc_email] += 1
        
        HistSentEmailNumGlobalUI[from_email] += 1

    # update HistReplyRateGlobalUI
    for key, val in HistRecEmailNumSTGlobalUI.items():
        HistReplyRateGlobalUI[key] += val
    for key, val in HistRecEmailNumCCGlobalUI.items():
        HistReplyRateGlobalUI[key] += val
    for key,val in HistReplyRateGlobalUI.items():
        HistReplyRateGlobalUI[key] = HistReplyNumGlobalUI[key]/val

    # update HistReplyTimeMeanGlobalUI and HistReplyTimeMedianGlobalUI
    for key, l in reply_time_list.items():
        mean_of_time, med_of_time = get_mean_med_time_list(l)
        
        # HistReplyTimeMeanGlobalUI
        HistReplyTimeMeanGlobalUI[key] = time_to_hours(mean_of_time)
        
        # HistReplyTimeMedianGlobalUI
        HistReplyTimeMedianGlobalUI[key] = time_to_hours(med_of_time)
    return HistReplyRateGlobalUI, HistReplyNumGlobalUI, HistRecEmailNumSTGlobalUI, HistRecEmailNumCCGlobalUI, HistSentEmailNumGlobalUI, HistReplyTimeMeanGlobalUI, HistReplyTimeMedianGlobalUI

def HistIndiv(email_item, res_HistIndiv):
    """
    Group: HistIndiv
    HistReplyRateGlobalUI
    HistReplyNumGlobalUI
    HistRecEmailNumSTGlobalUI
    HistRecEmailNumCCGlobalUI
    HistSentEmailNumGlobalUI
    HistReplyTimeMeanGlobalUI
    HistReplyTimeMedianGlobalUI
    HistGlobalUJ
    """
    from_email = email_item['From']
    HistReplyRateGlobalUI = res_HistIndiv[0][from_email]
    HistReplyNumGlobalUI = res_HistIndiv[1][from_email]
    HistRecEmailNumSTGlobalUI = res_HistIndiv[2][from_email]
    HistRecEmailNumCCGlobalUI = res_HistIndiv[3][from_email]
    HistSentEmailNumGlobalUI = res_HistIndiv[4][from_email]
    HistReplyTimeMeanGlobalUI = res_HistIndiv[5][from_email]
    HistReplyTimeMedianGlobalUI = res_HistIndiv[6][from_email]
    # HistGlobalUJ ignore
    return HistReplyRateGlobalUI, HistReplyNumGlobalUI, HistRecEmailNumSTGlobalUI, HistRecEmailNumCCGlobalUI, HistSentEmailNumGlobalUI, HistReplyTimeMeanGlobalUI, HistReplyTimeMedianGlobalUI 

# This is a pre_process one
def pre_HistPair(emails):
    """
    Group: HistPair
    HistReplyNumLocal
    HistReplyTimeMeanLocal
    HistReplyTimeMedianLocal
    """

    # HistReplyNumLocal
    HistReplyNumLocal = defaultdict(dict_int) # HistReplyNumLocal[u_i][u_j] u_i is sender, u_j is recipient
    # HistReplyTimeMeanLocal
    HistReplyTimeMeanLocal = defaultdict(dict_float) # HistReplyTimeMeanLocal[u_i][u_j] u_i is sender, u_j is recipient
    # HistReplyTimeMedianLocal
    HistReplyTimeMedianLocal = defaultdict(dict_float) # HistReplyTimeMedianLocal[u_i][u_j] u_i is sender, u_j is recipient
    waitinglist = defaultdict(dict_time) # waitinglist[w_i][w_j] = time -> w_j is waiting for the reply from w_i
    reply_time_list = defaultdict(dict_list) # reply_time_list[u_i][u_j] u_i is sender, u_j is recipient
    for i in range(len(emails)):
    # for i in range(500):
        try:
            email = emails.iloc[i]
        except:
            email = emails[i]
        from_email = email['From']
        # to_email may be nan
        try:
            to_emails = email['To'].split()
        except:
            to_emails = []
        
        # cc email not applied in Enr on
        try:
            cc_emails = email['CC'].split() # enron dosn't have cc, only X-cc
        except:
            cc_emails = []
        cur_time = parser.parse(email['Date'])
        
        for to_email in to_emails:
            waitinglist[to_email][from_email] = cur_time
            if to_email in waitinglist[from_email]: # it means from_email replied to_email
                # Frequency addd 1
                HistReplyNumLocal[to_email][from_email] += 1

                # preparation for ReplyTimeMean and Med
                # to_email (second key) sends msg to from_email (first key) at waitinglist[from_email][to_email]
                # I am quite sure that it is reply_time_list[to_email][from_email] (from_e reply/send to_e)
                reply_time_list[to_email][from_email].append(cur_time - waitinglist[from_email][to_email])
                
                # initialization
                del(waitinglist[from_email][to_email])
            
        for cc_email in cc_emails:
            waitinglist[cc_email][from_email] = cur_time
            if cc_email in waitinglist[from_email]: # it means from_email replied cc_email
                # Frequency addd 1
                HistReplyNumLocal[cc_email][from_email] += 1

                # preparation for ReplyTimeMean and Med
                # cc_email (second key) sends msg to from_email (first key) at waitinglist[from_email][cc_email]
                # I am quite sure that it is reply_time_list[cc_email][from_email]
                reply_time_list[cc_email][from_email].append(cur_time - waitinglist[from_email][cc_email])
                
                # initialization
                del(waitinglist[from_email][cc_email])

    for u_i in reply_time_list.keys():
        for u_j, reply_time_u_i_u_j in reply_time_list[u_i].items():
            mean_of_time, med_of_time = get_mean_med_time_list(reply_time_u_i_u_j)
        
            # HistReplyTimeMeanGlobalUI
            HistReplyTimeMeanLocal[u_i][u_j] = time_to_hours(mean_of_time)
            
            # HistReplyTimeMedianGlobalUI
            HistReplyTimeMedianLocal[u_i][u_j] = time_to_hours(med_of_time)
    return HistReplyNumLocal, HistReplyTimeMeanLocal, HistReplyTimeMedianLocal

def HistPair(email_item, res_HistPair):
    from_email = email_item['From']
    # to_email may be nan
    try:
        to_emails = email_item['To'].split()
    except:
        to_emails = []
    HistReplyNumLocal, HistReplyTimeMeanLocal, HistReplyTimeMedianLocal = [], [], []
    for to_email in to_emails:
        HistReplyNumLocal.append(res_HistPair[0][from_email][to_email])
        HistReplyTimeMeanLocal.append(res_HistPair[1][from_email][to_email])
        HistReplyTimeMedianLocal.append(res_HistPair[2][from_email][to_email])
    HistReplyNumLocal = get_mean_min_max(HistReplyNumLocal)
    HistReplyTimeMeanLocal = get_mean_min_max(HistReplyTimeMeanLocal)
    HistReplyTimeMedianLocal = get_mean_min_max(HistReplyTimeMedianLocal)
    return HistReplyNumLocal, HistReplyTimeMeanLocal, HistReplyTimeMedianLocal

def Meta(email_item):
    """
    Group: Meta
    HasAttachment
    NumOfAttachment
    """
    # Not available for Enron, available for Avocado
    # HasAttachment
    HasAttachment = 0 if len(email_item['Attachments']) == 0 else 1
    # NumOfAttachment
    NumOfAttachment = len(email_item['Attachments'])
    return HasAttachment, NumOfAttachment

def MetaAdded(email_item):
    """
    Group: MetaAdded
    IsImportant
    IsPriority
    IsSensitivity
    """
    # Not available for Enron, available for Avocado
    # IsImportant
    IsImportant = email_item['Important']
    # IsPriority
    IsPriority = email_item['Priority']
    # IsSensitivity
    IsSensitivity = email_item['Sensitivity']
    return IsImportant, IsPriority, IsSensitivity

def Temporal(email_item):
    """
    Group: Temporal
    TimeOfDay
    DayOfWeek
    WeekDayEnd
    """
    cur_time = parser.parse(email_item['Date'])
    
    # TimeOfDay
    TimeOfDay = [0] * 4
    intervals = [6, 12, 18, 24]
    for i, interval in enumerate(intervals):
        if cur_time.hour < interval:
            TimeOfDay[i] = 0
            break
    # DayOfWeek
    DayOfWeek = [0] * 7
    DayOfWeek[cur_time.weekday()] = 1
    
    # WeekDayEnd
    WeekDayEnd = [0] * 2
    if cur_time.weekday() < 5 : 
        WeekDayEnd[0] = 1
    else:
        WeekDayEnd[1] = 1
    
    return TimeOfDay, DayOfWeek, WeekDayEnd

def User(email_item):
    """
    Group: User
    UserDepartment
    UserJobTitle
    """
    # waiting to add, combine with job title level for enron
    # UserDepartment
    UserDepartment = email_item['Department']
    # UserJobTitle
    UserJobTitle = email_item['Stage']
    return UserDepartment, UserJobTitle
