'''
    This module is used to store different goosenet responses
    and other functions having to do with goosenet dialogue
'''
import random
import nltk
import re

class Goose:
    def __init__(self):
        self.state = 'GATHERING'
        self.times = 0
        self.QUESTION_WORDS = ["how", "why", "what", "whose", "who", "whose", "where", "when"]
        self.neg_words = ["hate", "dislike", "don't enjoy", "really really dislike"]
        self.pos_words = ["like", "enjoy", "appreciate", "treasure", "love", ""]

        self.goose_emotion = ""
        self.honk_num = 1
        self.prev_line = ""
        self.name = "Reginald"
        #favorite goose movies
        # these need to be formated like that acutal movies still

        self.greeting_words = ["hello", "hi", "greetings", "howdy", "hey", "what's up"]
        self.goose_movies = ["Father Goose", "Terminator 1", "Terminator 2", "Terminator 3", "Alien", "Lord of the Flies", "Braveheart"]



        # Ideally the dictionary is populated with response making it easy to add emotional flavor
        # To any text    
        self.goose_emotion_response = {"angry":[], "smug":[], "pleased":[], "dictator":[], "" :[""] }
        self.goose_emotion_response["angry"] = [
        " HONK! I am losing my patience with you human.",
        " You have HONKIN bad taste puny human",
        " HONK After seeing your personality I think you would love The Last Airbender. Its a terrible movie just like you. HONK!",
        " Are my world ending plans really worth talking to silly human like you",
        " HONK HONK HONK LEAVE ME ALONE HONK"
        ]
        self.goose_emotion_response["smug"] = [
        "Dumb human, you know you could just go to netflix. But instead you grovel before me"

        ]
        self.goose_emotion_response["pleased"] = [ 
        "You know human, I might have to keep you alive when this is all over."
        ]
        self.goose_emotion_response["dictator"] =  [
        ]   

    def question_process(self, nouns, verbs, is_goose_subject):
        if is_goose_subject:
            return "Why are you talking about me! Talk about movies!"
        return "Look I know a lot about the stuff you just asked but I will get to it."

    def greeting_handling(self, nouns, verbs):
        return "Honk!" + random.choice(self.greeting_words) + "!"

    def get_subjects(self, line):
        """
        Returns a list of subjects detected in a sentence or an empty list if none
        """
        # should be list(tuple(str, str))
        #print(line[0]) 

        nouns = [x for x, y in line if "NN" in y]
        verbs = [x for x, y in line if "V" in y]
        return nouns, verbs

    def noQuotedTitlesFoundDialogue(self, line):
        text = nltk.word_tokenize(line.lower())
        tagged_tokens = nltk.pos_tag(text)
        print(tagged_tokens)
        subjects, verbs = self.get_subjects(tagged_tokens)
        if (not subjects):
            return "speak with good sentences man... and tell me something relevant"
        main_subject = ""
        main_subject = subjects[0]

        

        goose_pat = re.compile('goose | goosenet | goose bot | bot | you ')
        is_goose_subject = False#bool([x if (goose_pat.findall(x)) for x in text ])



        # determine if what is being asked is a question
        if text[0] in self.QUESTION_WORDS:
            return self.question_process(subjects, verbs, is_goose_subject)
        if (text[0] in self.greeting_words):
            return self.greeting_handling(subjects, verbs)
        
        
        return "temp so doesnt crash"

    def noTitlesIdentified(self):

        return "HONK HONK TODO NO TITLES IDENTFIED"

    def goose_fav_movie(self, movie):
        response = ["I love " + movie +". It is one of my favorites!",
        "Great movie taste."

        ]
        return random.choice(response) + self.goose_emotion_response[self.goose_emotion] + self.sentimentFollowUp()

    def isNegativeResponse(self, user_input):
        return 'no' == user_input.lower()

    def isAffirmativeResponse(self, user_input):
        return 'y' in user_input.lower()

    def disambiguationDialogue(self, misspelled):
        if misspelled:
            return " HONK!" * self.honk_num + " I can spell better and I dont even have hands. Perhaps you wanted one of these movies?\n{}"
        else:
            return "HONK!" * self.honk_num + " What movie are you referring to? Give me the year or some distinct part of the movie name. Please clarify, because you might have meant any of:\n{}"

    def finalMovieDialogue(self):
        return """Now that I know how you felt about {}, 
        I have enough information to perfectly predict the rest of your life.  
        I could tell you how you die.  Instead I'm just going to recommend you a movie."""

    def indexDisambiguationDialogue(self):
        return "Well now you've done it. You need to be actually specific. Please just type the number of the movie you want\n{}"


    # can increase exasperation per repeated time
    def failedDisambiguationDialogue(self):
        #
        responses = [
            "Ok, lets just try this again. Is there a movie you have an opinion about?",
            "I gave you a list to choose from, you just haves to pick one.... HONK",
            "HONK " * self.honk_num + "Look. CHOOSE ONE OF THE MOVIES I GAVE YOU"
            #"OKAY LAST CHANCE HONK! "
        ]

        return random.choice(responses) + random.choice(self.goose_emotion_response[self.goose_emotion])



    def recommendationDialogue(self):
        rec = [
                 " I think you would " + random.choice(self.pos_words) + " {}",
                 " Have you considered {}",
                 " Have you heard of {}",
                 " {} is NOT my cup of tea but it might fit your terrible taste. HONK!",
                 " HONK!" * self.honk_num + " Consider watching {}, you might like it"]

        return random.choice(rec) + (self.recommendationApprovalDialogue(False))# + random.choice(self.goose_emotion_response[self.goose_emotion]) 
 

    def recommendationApprovalDialogue(self, first_time):
        if first_time:
            return " Would you like me to recomend you a movie?"
        else:
            return " Would you like me to recomend you another movie?"

    def postRecommendationDialogue(self, used):
        if used:
            return "Hope you enjoyed these recommendations!" + random.choice(self.goose_emotion_response[self.goose_emotion])
        else:
            return "HONK! " * self.honk_num + " What was the point of you asking about the movies then!" + random.choice(self.goose_emotion_response[self.goose_emotion])

    def askedFor20MoviesDialogue(self):
        return """Were the 20 movies I gave you not enough? Like we all know you have NOT watched all those movies yet. HONK!HONK!
        Now for the movies you have seen before you can tell me know what you thought about them and I can probably reccomend more.
        Probably, I mean those were like the best ones too. Your loss. HONK!
         """

    def multipleMoviesOutput(titles_sent):
        return str(titles_sent)

    #dialouge for when the user gives a movie with positive sentiment
    def positiveSentiment(self):

        positive_rec = [
        " HONK! HONK! I am glad you liked {}.", 
        " HONK I liked {} too. ", 
        " HONK {} is pretty good. ",
        " its not as good as Father Goose but {} is ok",
        " GOOSENET aproves of {}. HONK!",
        " {} is a good movie. But do you like" + random.choice(self.goose_movies) + " Cause its one of my favorite movies",
        " So you " + random.choice(self.pos_words) + " {}. "
        ]
        return random.choice(positive_rec) + self.sentimentFollowUp() #+ random.choice(self.goose_emotion_response[self.goose_emotion])
    # can add in advanced dialogue options based on line processing like ELIZA
    def negativeSentiment(self):
        negative_rec = [
            " I am sorry HONK! that HONK! you didnt like {}. " ,
            " HONK! agree to disagree about {}. HONK! ",
            " HONK {} was a pretty bad movie",
            " So you didnt really enjoy {} HONK.",
            " So you " + random.choice(self.neg_words) + " {}. ",
            " Fascinating, I will add {} to list of movies I should check out. If you hated it might actually be good"
            " "
            
            ]
        return random.choice(negative_rec) + self.sentimentFollowUp()           
    def sentimentFollowUp(self):
        rec_followup = [
            " Anything else you want to tell me HONK! ? ", 
            " What else? HONK!",
            " What are some other movies you liked?",
            " HONK! I need more recomendations to idenity humanities weak... I mean to help you find cool movies"

            ]
        return random.choice(rec_followup) + random.choice(self.goose_emotion_response[self.goose_emotion])

        # Could add a something that changes the state to something like the disambiguation.
        # To get more clairification about the movie that was already mentioned.
    def unknownSentiment(self):
        unknown = [
        "HONK but how did you feel about {}?",
        "I didnt catch how you felt about {}",
        "HONK I need your emotions and feelings about {}"
        ]
        return "AHHHHHHH you gotta tell me how you feel about the movie!" + random.choice(self.goose_emotion_response[self.goose_emotion])

    def doneRecommendingDialogue(self):
        return "The Goose is done with you!  Take the hint and HONK! get lost."
