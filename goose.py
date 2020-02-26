'''
    This module is used to store different goosenet responses
    and other functions having to do with goosenet dialogue
'''
import random

class Goose:
    def __init__(self):
        self.state = 'GATHERING'
        self.times = 0

    def noQuotedTitlesFoundDialogue(self):
        return "I am a Goose on a mission.  If you're not talking movies or US supply lines, I don't want to talk."

    def disambiguationDialogue(self):
        return "HONK! What movie are you reffering to? I found these movies {}."

    def noTitlesIdentified(self):
        return "HONK TO DO HONK I GOT NO CLUE WHAT YOU ARE TALKING ABOUT"

    def misspelled(self):
        return " HONK I can spell better and I dont even have hands. Perhaps you wanted one of these movies? {} HONK!"

    def recommendationDialogue(self):
        return " I think you would like {}"

    def recommendationApprovalDialogue(self, first_time):
        if first_time:
            return "Would you like me to recomend you a movie?"
        else:
            return "Would you like me to recomend you another movie?"
    