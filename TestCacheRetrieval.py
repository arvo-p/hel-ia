from CacheManager import CacheManager
from Tools import *

cache_manager = CacheManager()
while True:
    question = input("Question: ")
    question_aftertreatment = Tools.clean_query(question)
    print("TREATED: " + question_aftertreatment)
    ret = cache_manager.search(question_aftertreatment)
    if ret is not None:
        print(ret[0])
