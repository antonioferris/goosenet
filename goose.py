'''
    This module is used to store different goosenet responses
    and other functions having to do with goosenet dialogue
'''
import random

class Goose:
    def __init__(self):
        self.state = 'GATHERING'
        self.times = 0
        self.QUESTION_WORDS = ["how", "why", "what", "whose", "who", "whose", "where", "when"]
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

    def indexDisambiguationDialogue(self):
        return "Well now you've done it.  You need to be actually specific.  Please just type the number of the movie you want\n{}"


    # can increase exasperation per repeated time
    def failedDisambiguationDialogue(self):
        #goose exaperation += 1
        rec = [
            "Ok, lets just try this again.  Is there a movie you have an opinion about?",
            "I gave you a list to choose from you just got to pick one.... HONK"
            "HONK HONK HONK HONK HONK HONK HONK HONK HONK Look. CHOOSE ONE OF THE MOVIES I GAVE YOU"
            "OKAY LAST CHANCE HONK! "
        ]

        return 

    def noTitlesIdentified(self):
        return "HONK TO DO HONK I GOT NO CLUE WHAT YOU ARE TALKING ABOUT"

    def recommendationDialogue(self):
            rec = [
                 " I think you would like {}",
                 " Have you considered {}",
                 "Have you heard of {}",
                 "{} is NOT my cup of tea but it might fir your terrible taste. HONK!",
                 "HONK! Consider watching {}, you might like it"

            ]
        return random.choice(rec)

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

    def askedFor20MoviesDialogue(self):
        return """Were the 20 movies I gave you not enough? Like we all know you have NOT watched those movies yet. HONK!HONK!
        Now for the movies you have seen before you can tell me know what you thought about then and I can probably reccomend more.
        Probably, I mean those were like the best ones too. Your loss. HONK!
         """
    #dialouge for when the user gives a movie with positive sentiment
    def positiveSentiment(self):

        positive_rec = [
        " HONK! HONK! I am glad you liked {}. ", 
        " HONK I liked {} too. ", 
        "HONK {}. is pretty good. "
        ]
        return random.choice(positive_rec)
    # can add in advanced dialogue options based on line processing like ELIZA
    def negativeSentiment(self):
        negative_rec = [
            "I am sorry HONK! that HONK! you didnt like {}. " ,
            "HONK! agree to disagree about {}. HONK! ",
            "HONK I LOVED {}, so you know what? HONK YOU!",
            "So you didnt really enjoy {} HONK."


            ]
        return random.choice(negative_rec)                
    def sentimentFollowUp(self):
        rec_followup = [
            "Anything else you want to tell me HONK! ? ", 
            " What else? HONK!",
            "What are some other movies you liked?",
            "HONK! I need more recomendations to idenity humanities weak... I mean to help you find cool movies"

            ]
        return random.choice(rec_followup)

        # Could add a something that changes the state to something like the disambiguation.
        # To get more clairification about the movie that was already mentioned.
    def unknownSentiment(self):
        unknown = [
        "HONK but how did you feel about {}?",
        "I didnt catch how you felt about {}",
        "HONK I need your emotions and feelings about {}"
        ]
        return "AHHHHHHH"
