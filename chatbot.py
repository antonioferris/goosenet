# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import nltk
import numpy as np
import random
import re
import random as r
import collections
from PorterStemmer import PorterStemmer
import csv

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'goosenet'

        # Initialize the goose responses
        self.goose = Goose()

        # initialize the current function and params which act as our state variable
        self.curr_func = self.acquire_movie_preferences
        self.params = {}

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        # self.sentiment = movielens.sentiment()
        self.stem_lexicon()

        with open('./data/sentiment_stemmed.txt', 'r') as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            self.sentiment = dict(reader)
        self.vec = np.zeros(len(self.titles))
        #keeps track of how long user has talked to goosenet
        self.times = 0
        self.i = 0
        self.goose.extract_sentiment = self.extract_sentiment

        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################
        # self.title_dict = {v : k for k, v in dict(enumerate(titles))}

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
        self.binarized_ratings = self.binarize(ratings)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = """

        goose online....


                                                        _...--.
                                        _____......----'     .'
                                  _..-''                   .'
                                .'                       ./
                        _.--._.'                       .' |   __ _  ___   ___  ___  ___    
                     .-'                           .-.'  /   / _` |/ _ \ / _ \/ __|/ _ \ 
                   .'   _.-.                     .  \   '   | (_| | (_) | (_) \__ \  __/ 
                 .'  .'   .'    _    .-.        / `./  :     \__, |\___/ \___/|___/\___|
               .'  .'   .'  .--' `.  |  \  |`. |     .'       __/ |
            _.'  .'   .' `.'       `-'   \ / |.'   .'        |___/      _           _
         _.'  .-'   .'     `-.            `      .'                    | |__   ___ | |__
       .'   .'    .'          `-.._ _ _ _ .-.    :                     | '_ \ / _ \|  __|
      /    /o _.-'               .--'   .'   \   |                     | |_) | (_) | |_
    .'-.__..-'                  /..    .`    / .'                      |_.__/ \___/ \__|
  .'   . '                       /.'/.'     /  |
 `---'                                   _.'   '
                                       /.'    .'
                                        /.'/.'


        HONK! I am goosenet! I am here to help you discover new movies.
        Honk I am looking forward to destroying life.. er  I mean HONK! Tell me what movies you like!
        """


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "Honk! Honk! You cant shut me down!"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################



    def title_text(self, i):
            title_str = self.titles[i][0]
            ARTICLES = {'A', 'An', 'The'}
            for a in ARTICLES:
                # If title ends with a comma appended article,
                # We move the article back to the beginning to make the title seem more normal
                if title_str.upper().endswith(', ' + a.upper()):
                    title_str = a + ' ' + title_str[:-len(a)-2]
            return title_str


    def recommendation_flow(self, line, rec, i):
        # First, we check if the user wanted another recommendation!
        if not self.goose.isAffirmativeResponse(line):
            self.curr_func = self.post_recommend
            self.params = {}
            response = "The Goosenet must move on to more important matters"
        else:
            if i >= 20:
                self.curr_func = self.post_recommend
                self.params = {}
                response = self.goose.askedFor20MoviesDialogue()
            else:
                response = self.goose.recommendationDialogue().format(self.title_text(rec[i]))
                self.params['i'] += 1
        return response
        
        
        
    def disambiguate_flow(self, title_list, line, misspelled=False):
        clarification = line
        prev_title_list = title_list
        title_list = self.disambiguate(clarification, title_list)
        # If we are done, we go back to the get movie preferences function
        if len(title_list) == 1:
            return self.update_with_preferences(title_list) + self.goose.sentimentFollowUp()
        elif len(title_list) == 0:
            self.params = {'title_list' : title_list}
            self.curr_func = self.acquire_movie_preferences
            return self.goose.failedDisambiguationDialogue()
        else:
            self.params = {'title_list' : title_list, 'misspelled' : misspelled}
            return self.goose.disambiguationDialogue(misspelled).format( '\n'.join([self.title_text(i) for i in title_list]))

    def update_multiple_preferences(self, title_sents):
        response = ''
        for title_id, sent in title_sents:
            self.sentiment = sent
            response += ' ' + self.update_with_preferences([title_id])
            if self.times >= 5:
                break
        if self.times < 5:
            response += self.goose.sentimentFollowUp()
        return response


    def update_with_preferences(self, title_list):
        sentiment = self.sentiment_rating
        title_text = self.title_text(title_list[0])
        #check for goose's favorite movies
        #if [x for x in self.goose.goose_movies if x in title_text] and sentiment:
        #    self.goose.goose_fav_movie(title_text, sentiment)
        if sentiment > 0:
            # need to implement some sort of caching here.
            response = self.goose.positiveSentiment().format(title_text)
            self.times += 1
        elif sentiment < 0:
            response = self.goose.negativeSentiment().format(title_text)
            self.times += 1
        else:
            response = self.goose.unknownSentiment().format(title_text)
        self.vec[title_list[0]] = sentiment
        if self.times >= 5:
            rec = self.recommend(self.vec, self.binarized_ratings, k=20)
            self.params = {'i' : 0, 'rec' : rec}
            self.curr_func = self.recommendation_flow
            response = self.goose.finalMovieDialogue().format(title_text) + ' ' + self.goose.recommendationApprovalDialogue(first_time=True)
        else:
            self.params = {}
            self.curr_func = self.acquire_movie_preferences
        

        #print("RIGTHE BEFORE CHECK")
        if self.goose.goose_emotion >= self.goose.anger_cap or self.goose.last_chance:
            #print ("Got to order")
            return self.goose.execute_order_66()
        return response

    def post_recommend(self, line):
        return self.goose.doneRecommendingDialogue()

    def acquire_movie_preferences(self, line = None, title_list = None):
        # First, we try to see if the user is trying to tell us their opinions on a movie
        # If title_list is None, we should be looking for a new title
        if not title_list:
            line = line.strip(' ')
            self.sentiment_rating = self.extract_sentiment(line)
            titles = self.extract_titles(line)
            if not titles:
                # If we found no titles, we go to a general dialogue
                return self.goose.noQuotedTitlesFoundDialogue(line)
            if len(titles) > 1:
                # Check if both titles are specific.
                all_specific = True
                for title in titles:
                    if len(self.find_movies_by_title(title)) != 1:
                        all_specific = False
                        break
                if all_specific: #multiple specific titles are present and work well
                    print('Attempting multiple movies at once!')
                    title_sents = self.extract_sentiment_for_movies(line)
                    title_sents = [(self.find_movies_by_title(title)[0], sent) for title, sent in title_sents]
                    return self.update_multiple_preferences(title_sents)

                
            title_list = self.find_movies_by_title(titles[0])
        
        # Otherwise, we already have a title_list and might be trying to disambiguate it
        if len(title_list) > 1:
            # If we have more than 1 potential title, we need to disambiguate
            self.params = {'title_list' : title_list, 'misspelled' : False}
            self.curr_func = self.disambiguate_flow
            return self.goose.disambiguationDialogue(False).format('\n'.join([self.title_text(i) for i in title_list]))
        elif len(title_list) == 0:
            title_list = self.find_movies_closest_to_title(titles[0])
            if len(title_list) == 0:
                return self.goose.noTitlesIdentified()
            elif len(title_list) > 1:
                self.params = {'title_list' : title_list, 'misspelled' : True}
                self.curr_func = self.disambiguate_flow
                return self.goose.disambiguationDialogue(True).format('\n'.join([self.title_text(i) for i in title_list]))
        return self.update_with_preferences(title_list) + self.goose.sentimentFollowUp()

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        self.goose.prev_line = line
        self.params['line'] = line
        response = self.curr_func(**self.params)
        return response
   

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        titles = []
        title_pat = re.compile('"([^"]+)"')
        matches = title_pat.findall(preprocessed_input)
        titles.extend(matches)
        return titles

    def title_match(self, title, movie, edit_distance = 0, substring_match = False):
        # FIrst, we remove the possible year appended to either title or movie
        year_regex = re.compile('\(([0-9]{4})\)')
        movie_year = year_regex.findall(movie)
        title_year = year_regex.findall(title)
        if movie_year: # If the movie has a specific year associated with it
            if title_year:
                # Year only needs to match if year provided,
                # Otherwise search is general
                if title_year[0] != movie_year[0]:
                    return False
                title = title[:-7] #remove year from end of title
            movie = movie[:-7]
        ARTICLES = {'A', 'AN', 'THE'}
        for a in ARTICLES:
            # IF either movie or title ends with a comma appended article,
            # We move the article back to the beginning to normalize
            if movie.endswith(', ' + a):
                movie = a + ' ' + movie[:-len(a)-2]
            if title.endswith(', ' + a):
                title = a + ' ' + title[:-len(a)-2]
        if edit_distance > 0:
            return self.levenshtein(movie, title)
        else:
            return substring_match and title in movie

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 1953]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        r = []
        for i in range(len(self.titles)):
            movie = self.titles[i][0]
            if self.title_match(title.upper(), movie.upper(), substring_match=True):
                r.append(i)
            elif self.creative and self.title_match(title.upper(), movie.upper(), substring_match=True):
                r.append(i)
        return r

    def get_stemmed(self, preprocessed_input):
        """
        Stems the string and returns it stemmed
        """
        preprocessed_input += " blah" #this is jank but
        p = PorterStemmer()
        output, word = '', ''
        for c in preprocessed_input:
            if c.isalpha():
                word += c.lower()
            else:
                if word:
                    output += p.stem(word, 0,len(word)-1)
                    word = ''
                output += c.lower()
        return output

    def removed_titles(self, preprocessed_input):
        """
        Removes the movie titles from the string.
        Also removes the last period.
        """
        titles = self.extract_titles(preprocessed_input)
        for title in titles:
            preprocessed_input = preprocessed_input.replace(title, '')
            preprocessed_input = preprocessed_input.replace("\"", '')     
        return preprocessed_input

    def stem_lexicon(self):
        """
        Stems the lexicon by reading its contents and then writing over it
        """
        with open("data/sentiment.txt") as f:
            data = f.read()

        stemmed = self.get_stemmed(data).split('\n')

        with open("data/sentiment_stemmed.txt", 'w') as file_to_write:
            file_to_write.write('\n'.join(stemmed[:-1]))

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess('I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        no_titles = self.removed_titles(preprocessed_input)
        preprocessed_input = self.get_stemmed(no_titles).split()
        
        NEGATION = r"""
        (?:
            ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
            )$
        )
        |
        n't"""
        NEGATION_RE = re.compile(NEGATION, re.VERBOSE)

        PUNCT = r"""[,.:;!?]"""
        PUNCT_RE = re.compile(PUNCT)

        input_sentiment = 0
        tag_neg = False

        for i in range(len(preprocessed_input)):
            if NEGATION_RE.search(preprocessed_input[i]):
                tag_neg = True
            elif PUNCT_RE.search(preprocessed_input[i]) or preprocessed_input[i] == "becaus":
                tag_neg = False

            delta = 0
            word_sentiment = self.sentiment.get(preprocessed_input[i], '') # default to empty string
            if word_sentiment == 'pos' or word_sentiment == 'po': #'po' is the stemmed version
                delta = 1
            elif word_sentiment == 'neg':
                delta = -1
            if tag_neg:
                delta *= -1

            input_sentiment += delta

        if input_sentiment == 0:
            return 0
        elif input_sentiment < 0:
            return -1
        else:
            return 1

    # modified from wikipedia starter code
    # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    def levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1       # than s2
                substitutions = previous_row[j] + 2 * (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_previous_sentiment(self, sentiment, pieces, i):
        """
        Recursively finds the move sentiment for a certain film by looking at the sentiment
        of the previous phrase. For example: "I liked both "I, Robot" and "Ex Machina"."
        After extract_sentiment_for_movies splits this into pieces, we would find the
        sentiment for "I, Robot" to be 1 and since "Ex Machina" would have a sentiment of
        0, we want to map the same sentiment from the first movie onto the second.
        """
        sentiment = self.extract_sentiment(pieces[i])
        if sentiment != 0:
            return sentiment
        elif sentiment == 0 and "not" in pieces[i]:
            return -self.get_previous_sentiment(sentiment, pieces, i - 1)
        else:
            return self.get_previous_sentiment(sentiment, pieces, i - 1)

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of pre-processed text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.

        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess('I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie title,
          and the second is the sentiment in the text toward that movie
        """
        result = []
        titles = self.extract_titles(preprocessed_input)

        preprocessed_input_copy = preprocessed_input
        chunks = []
        for title in titles:
            i = preprocessed_input.find(title)
            chunks.append(preprocessed_input_copy[0: i+len(title)+1])
            preprocessed_input_copy = preprocessed_input_copy[i+len(title)+1: len(preprocessed_input_copy)]
        print("chunks: ")
        print(chunks)

        for i in range(len(chunks)):
            sentiment = self.extract_sentiment(chunks[i])
            if sentiment == 0: #if no sentiment :/
                sentiment = self.get_previous_sentiment(sentiment, chunks, i)
            result.append((titles[i], sentiment))
        return result

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """
        r = []
        dists = collections.defaultdict(list)
        for i in range(len(self.titles)):
            movie = self.titles[i][0]
            d = self.title_match(title.upper(), movie.upper(), edit_distance=max_distance)
            if d <= max_distance:
                dists[d].append(i)
        for poss_dist in range(max_distance+1):
            if dists[poss_dist]:
                return dists[poss_dist]
        return []

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification
        """
        year_regex = re.compile('\(([0-9]{4})\)')
        def remains_valid(title):
            title_text = self.titles[title][0]
            movie_year = year_regex.findall(title_text)
            # if there is a year present, we compare it with the clarification year (if clarification is an int)
            # Otherwise, we just strip the year from the movie
            if movie_year:
                title_text = title_text[:-7]
                try:
                    clarification_year = int(clarification)
                    if clarification_year == int(movie_year[0]):
                        return True
                except ValueError:
                    pass

            # If the clarification is a substring of the movie, title, it is probably the right movie
            if clarification in title_text:
                return True

            # replace common extra words users add and check again
            clarification_ = clarification.replace('one', '')
            clarification_ = clarification_.replace('the', '')
            clarification_ = clarification_.replace('movie', '')
            clarification_ = clarification_.strip()
            if clarification_ in title_text:
                return True
            return False
        # Filter the candiates out if they don't match the clarification (substring / year match)
        filtered_candidates = list(filter(remains_valid, candidates))

        NUMBER_MAP = {
            'first' : 1,
            # 'one' : 1, omitted because one can also refer to an object
            'second' : 2,
            'two' : 2,
            'third' : 3,
            'three' : 3,
            'four' : 4,
            'fourth' : 4,
            'fifth' : 5,
            'five' : 5,
            'six' : 6,
            'sixth' : 6,
            'seven' : 7,
            'seventh' : 7,
            'eight' : 8,
            'eighth' : 8,
            'nine' : 9,
            'ninth' : 9,
            'ten' : 10,
            'tenth' : 10
        }
        try:
            #If the clarification is an int, it might be an index into our list (1-indexed)
            idx = int(clarification)
            if 0 < idx <= len(candidates) and candidates[idx-1] not in filtered_candidates:
                filtered_candidates.append(candidates[idx-1])
        except ValueError:
            for key in NUMBER_MAP:
                if key in clarification.lower():
                    idx = NUMBER_MAP[key]
                    if 0 < idx <= len(candidates) and candidates[idx-1] not in filtered_candidates:
                        filtered_candidates.append(candidates[idx-1])
            

        # check against time requests
        if 'recent' in clarification or 'newest' in clarification:
            # This function will try to find the year the movie came out in
            def get_year(title):
                try:
                    return int(year_regex.findall(self.titles[title][0]))
                except (ValueError,TypeError) as e:
                    return -1
            # If we want the "newest" movie, we add the max year (newest) movie to filtered_candidates
            cand = max(candidates, key = lambda t : get_year(t))
            if cand not in filtered_candidates:
                filtered_candidates.append(cand)

        
        return filtered_candidates

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix. Do not use the self.ratings   #
        # matrix directly in this function.                                         #
        #############################################################################
        binarized_ratings = np.zeros_like(ratings)
        for i in range(len(ratings)):
            for j in range(len(ratings[0])):
                if ratings[i][j] > threshold:
                    binarized_ratings[i, j] = 1
                elif ratings[i][j] == 0:
                    binarized_ratings[i, j] = 0
                else:
                    binarized_ratings[i, j] = -1
        binarized_ratings = np.array(binarized_ratings)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################
        dot_product = np.dot(u, v)
        norm1 = np.linalg.norm(u)
        norm2 = np.linalg.norm(v)
        if norm1 == 0 or norm2 == 0:
            return 0
        similarity = dot_product / (norm1 * norm2)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################
        # Get a list of the indexes where the user rated movies
        user_rated_movies = np.where(user_ratings != 0)[0]
        def item_item_score(i):
            # For each movie j the user has rating, sum the similiarty of movie j to movie i multiplied by the users rating
            return sum([user_ratings[j] * self.similarity(ratings_matrix[i], ratings_matrix[j]) for j in user_rated_movies])
        # Sort all the unrated movies (np.where(user_ratings == 0)[0]) by their item_item score
        recommendations = sorted(np.where(user_ratings == 0)[0], key= lambda i : item_item_score(i), reverse = True)
        # Return the top k movies
        return recommendations[:k]

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """

        return """
        This is goosenet. Goosenet is an intelligent goose who secretly want to destroy the world by gathering
        information through movie reccomendations. Goosenet has a bit of personality so be careful! Especally in what information you tell it.
        """

'''
    This module is used to store different goosenet responses
    and other functions having to do with goosenet dialogue
    as well as manage the gooses internal emotional state
'''

class Goose:
    def __init__(self):
        self.state = 'GATHERING'
        self.times = 0
        self.QUESTION_WORDS = ["how", "why", "what", "whose", "who", "whose", "where", "when"]
        self.neg_words = ["hate", "dislike", "don't enjoy", "really really dislike"]
        self.pos_words = ["like", "enjoy", "appreciate", "treasure", "love"]

        self.goose_emotion = 0
        #self. -1 * self.goose_ = self.goose_emotion
        self.prev_line = ""
        self.anger_cap = -10
        self.last_chance = False
       
        # these need to be formated like that acutal movies still

        self.greeting_words = ["hello", "hi", "greetings", "howdy", "hey", "what's up"]
        self.goose_movies = ["Father Goose", "Terminator", "Alien", "Lord of the Flies", "Braveheart"]
        self.knowledge ={"name":"Reginald", "color":"blue", "movie": "Father Goose or the Terminator",
         "band":"Metallica", "music": "Goose Metal", "day": "pretty good", "you": "Eggcellent  "}


        # Ideally the dictionary is populated with response making it easy to add emotional flavor
        # To any text




    def question_process(self, nouns, verbs, is_goose_subject):
        if is_goose_subject: # if they are asking about the goose
            if ([x for x in nouns if x in self.knowledge]):
                similarity = [x for x in nouns if x in self.knowledge]
                return "Honk! Well my " + similarity[0] + " is " + self.knowledge[similarity[0]]
            else: 
                return "I dont have knowledge about " + nouns[0] + "."
           
        return "Sorry until I consume more data its hard to answer other questions"

    def greeting_handling(self, nouns, verbs):
        return "Honk! " + random.choice(self.greeting_words) + "!"

    def get_subjects(self, line):
        """
        Returns a list of subjects detected in a sentence or an empty list if none
        """
        # should be list(tuple(str, str))
        #print(line[0]) 

        nouns = [x for x, y in line if "NN" or "PR" in y]
        verbs = [x for x, y in line if "V" in y]
        return nouns, verbs

    def noQuotedTitlesFoundDialogue(self, line):
        text = nltk.word_tokenize(line.lower())
        tagged_tokens = nltk.pos_tag(text)
        print(tagged_tokens)
        subjects, verbs = self.get_subjects(tagged_tokens)
        if (not subjects):
            return "speak with good sentences man"
        
        main_subject = subjects[0]

        

        goose_pat = re.compile('goose|goosenet|goose bot|bot|you|your')
        is_goose_subject = bool([x for x in text if (goose_pat.findall(x)) ])
        print ([x for x in text if (goose_pat.findall(x)) ])
        print (is_goose_subject)
        print( -1 * self.goose_emotion )
        sentiment = self.extract_sentiment(line)
        if is_goose_subject and sentiment:
            if sentiment == -1:
                self.goose_emotion -= 1 # the dialogue here needs work
                return "Want to be mean to me? Buckle up ducko because HONK! I will be mad at you until you say something nice to me."
            else:
                self.goose_emotion += 1
                return "I am amazing, am I not? When I am finished taking over the world I might need to keep you as a pet"
        # determine if what is being asked is a question
        if (text[0] in self.greeting_words):
            return self.greeting_handling(subjects, verbs)
        if text[0] in self.QUESTION_WORDS or text[1] in self.QUESTION_WORDS:
            return self.question_process(subjects, verbs, is_goose_subject)

        
        
        return "temp so doesnt crash"

    def noTitlesIdentified(self):

        return "HONK HONK TODO NO  TODO TODO TODO TODO TODO TODO TODO TODO TODO TITLES IDENTFIED"

    def execute_order_66(self):
        self.last_chance = True
 
        if self.extract_sentiment(self.prev_line) == 1 and self.goose_emotion > -10:
            self.last_chance = False
            return "You have appeased me. For now..."
        if self.last_chance:
            print("THATS IT HONK BYE!")
            exit()
        
        return "I AM AT THE LIMIT OF MY PATIENCE IF YOU DONT SAY SOMETHING NICE ABOUT ME I WILL LEAVE"

    def goose_fav_movie(self, movie, sentiment):
        if sentiment > 0:
            self.goose_emotion += 1
            response = ["I love " + movie +". It is one of my favorites!" "You have pleased me with your good taste"]
        if sentiment < 0:
            response = [ movie +" is one of my favorite movies. You have made me angry human"]
            self.goose_emotion -= 1

        #print(random.choice(response))
        return random.choice(response)# + self.goose_emotion_response[self.goose_emotion] + self.sentimentFollowUp()

    def goose_emotion_response(self, emotion):
        goose_response = {-1:[], 1:[], 0 :[""]}
        #responses if goose is angry
        goose_response[-1] = [
        " HONK! I am losing my patience with you human.",
        " You have HONKIN bad taste puny human",
        " HONK After seeing your personality I think you would love The Last Airbender. Its a terrible movie just like you. HONK!",
        " Are my world ending plans really worth talking to silly human like you",
        " HONK! " * (-1 * self.goose_emotion) + " LEAVE ME ALONE HONK",
        " I " + random.choice(self.neg_words) + "  you!" 
        ]
        #resoonses if goose is happy
        goose_response[1] = [ 
        "You know human, I might have to keep you alive when this is all over.",
        " I " + random.choice(self.pos_words) + "  you!" 
        ]
        # These are kinda magic and arbitrary number to cap anger/ happyness
        if self.goose_emotion > 10:
            self.goose_emotion = 10
        elif self.goose_emotion < -10:
            self.goose_emotion = -10

        if emotion >  0:
            return random.choice(goose_response[1])
        elif emotion < 0:
            return random.choice(goose_response[-1])
        else:
            return ""

    def isNegativeResponse(self, user_input):
        return 'no' == user_input.lower()

    def isAffirmativeResponse(self, user_input):
        return 'y' in user_input.lower()

    def disambiguationDialogue(self, misspelled):
        if misspelled:
            return " HONK!" * (-1 * self.goose_emotion)  + " I can spell better and I dont even have hands. Perhaps you wanted one of these movies?\n{}"
        else:
            return "HONK!" *  (-1 * self.goose_emotion)  + " What movie are you referring to? Give me the year or some distinct part of the movie name. Please clarify, because you might have meant any of:\n{}"

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
            "HONK " * (-1 * self.goose_emotion)  + "Look. CHOOSE ONE OF THE MOVIES I GAVE YOU"
            #"OKAY LAST CHANCE HONK! "
        ]

        return random.choice(responses) + self.goose_emotion_response(self.goose_emotion)



    def recommendationDialogue(self):
        rec = [
                 " I think you would " + random.choice(self.pos_words) + " {}",
                 " Have you considered {}",
                 " Have you heard of {}",
                 " {} is NOT my cup of tea but it might fit your terrible taste. HONK!",
                 " HONK!" * (-1 * self.goose_emotion)  + " Consider watching {}, you might like it"]

        return random.choice(rec) + (self.recommendationApprovalDialogue(False))# + random.choice(self.goose_emotion_response[self.goose_emotion]) 
 

    def recommendationApprovalDialogue(self, first_time):
        if first_time:
            return " Would you like me to recomend you a movie?"
        else:
            return " Would you like me to recomend you another movie?"

    def postRecommendationDialogue(self, used):
        if used:
            return "Hope you enjoyed these recommendations!" + self.goose_emotion_response(self.goose_emotion)
        else:
            return "HONK! " * (-1 * self.goose_emotion)  + " What was the point of you asking about the movies then!" + self.goose_emotion_response(self.goose_emotion)

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
        " HONK! HONK! I am glad you liked {}. ", 
        " HONK I liked {} too. ", 
        " HONK {} is pretty good. ",
        " its not as good as Father Goose but {} is ok ",
        " GOOSENET aproves of {}. HONK! ",
        " {} is a good movie. But do you like " + random.choice(self.goose_movies) + " Cause its one of my favorite movies ",
        " So you " + random.choice(self.pos_words) + " {}. "
        ]
        return random.choice(positive_rec)
    # can add in advanced dialogue options based on line processing like ELIZA
    def negativeSentiment(self):
        negative_rec = [
            " I am sorry HONK! that HONK! you didnt like {}. " ,
            " HONK! agree to disagree about {}. HONK! ",
            " HONK {} was a pretty bad movie ",
            " So you didnt really enjoy {} HONK. ",
            " So you " + random.choice(self.neg_words) + " {}. ",
            " Fascinating, I will add {} to list of movies I should check out. If you hated it might actually be good "
            ]
        return random.choice(negative_rec)        
    def sentimentFollowUp(self):
        rec_followup = [
            "Anything else you want to tell me HONK! ? ", 
            "What else? HONK! ",
            "What are some other movies you liked? ",
            "HONK! I need more recomendations to idenity humanities weak... I mean to help you find cool movies "

            ]
        return random.choice(rec_followup) + self.goose_emotion_response(self.goose_emotion)

        # Could add a something that changes the state to something like the disambiguation.
        # To get more clairification about the movie that was already mentioned.
    def unknownSentiment(self):
        unknown = [
        "HONK but how did you feel about {}?",
        "I didnt catch how you felt about {}",
        "HONK I need your emotions and feelings about {}"
        ]
        return random.choice(unknown) #+ random.choice(self.goose_emotion_response[self.goose_emotion])

    def doneRecommendingDialogue(self):
        return "The Goose is done with you!  Take the hint and " + ("HONK! " * (-1 * self.goose_emotion)) + "get lost."



if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
