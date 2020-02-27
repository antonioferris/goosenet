'''
    This module is used to store different goosenet responses
    and other functions having to do with goosenet dialogue
'''
import random

class Goose:
    def __init__(self):
        self.state = 'GATHERING'
        self.times = 0

    def isNegativeResponse(self, user_input):
        return 'n' in user_input.lower()

    def isAffirmativeResponse(self, user_input):
        return 'y' in user_input.lower()

    def noQuotedTitlesFoundDialogue(self):
        return "I am a Goose on a mission.  If you're not talking movies or US supply lines, I don't want to talk."

    def disambiguationDialogue(self, misspelled):
        if misspelled:
            return " HONK I can spell better and I dont even have hands. Perhaps you wanted one of these movies?\n{}"
        else:
            return "HONK! What movie are you referring to?  Please clarify, because you might have meant any of:\n{}"

    def overDisambiguatedDialogue(self)

    def noTitlesIdentified(self):
        return "HONK TO DO HONK I GOT NO CLUE WHAT YOU ARE TALKING ABOUT"

    def recommendationDialogue(self):
        return " I think you would like {}"

    def recommendationApprovalDialogue(self, first_time):
        if first_time:
            return "Would you like me to recomend you a movie?"
        else:
            return "Would you like me to recomend you another movie?"

    def postRecommendationDialogue(self, used):
        if used:
            return "Hope you enjoyed these recommendations!"
        else:
            return "HONK!  What was the point of you asking about the movies then!"

    def askedFor20MoviesDialgoue(self):
        return "Were the 20 movies I gave you not enough?  Let me know what you thought of them and I can recommend more"
    