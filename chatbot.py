# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import nltk
import numpy as np
import re
import random as r

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'goosenet'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.sentiment = movielens.sentiment()
        self.vec = np.zeros(len(self.titles))
        #keeps track of how long user has talked to goosenet
        self.times = 0

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


        HONK! I am goosenet! I am here to help you discover new movies in return for sensitive information about
        US troop distributions and supply lines! I give you bonus recomendations for goose related things!
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

    def get_subjects(self, line):
        """
        Returns a list of subjects detected in a sentence or an empty list if none
        """
        
        words = [ i[0] for i in line if 'N' in i[1]] 
        return words


    def question_process(self, line, tagged_tokens):
        subjects = self.get_subjects(tagged_tokens)
        return (" HONK HONK I lNOW ALL about {}. BUT DONT TELL.".format(subjects[0]))

    def loop_rec(self):
        line = input("Would you like me to reccomend you a movie?")
        rec =  self.recommend(self.vec, self.binarized_ratings)
        i = 0
        # logic not correct but bleh rn
        while(line != "no" ):
            
            #print (" SELF VEC", self.vec)
            #print ("\n RATINGS", self.binarized_ratings)

            #print(" BEFORE RECCOMENDING")
            i += 1
            print(" I think you would like {}".format((self.titles[rec[i]][0])))
            line = input("Would you like me to reccomend you another movie?")

            #print("AFTER RECCOMEDING", reccomendations)
        

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
        QUESTION_WORDS = ["how", "why", "what", "whose", "who", "whose", "where", "when"]

        list_of_words = nltk.word_tokenize(line)
        tagged_tokens = nltk.pos_tag(list_of_words)
        #print (type(nltk.chunk.ne_chunk(tagged_tokens)))

        #if any(map(line.lower().startswith, QUESTION_WORDS)):
            #print (tagged_tokens)
    
            #return self.question_process(line, tagged_tokens)


        

        positive_rec = [" HONK! HONK! I am glad you liked {}. ", " HONK I liked {} too. ", "HONK {}. is pretty good. "]
        negative_rec = ["I am sorry HONK! that HONK! you didnt like {}. " ,"HONK! agree to disagree about {}. HONK! "]
        rec_followup = ["Anything else you want to tell me HONK! ? " , " What else HONK!"]
        unknown_rec = ["I didnt catch your thoughts on {}. HONK! ", "Can you tell me more about your thoughts on {}. HONK? " ]
        goose_specific = ["GOOSENET aprooves "]



        followup = r.choice(rec_followup)

        sentiment = self.extract_sentiment(line)
        titles = self.extract_titles(line)
        title_list = self.find_movies_by_title(titles[0])
     

        if len(title_list) > 1:
            return "HONK! What movie are you reffering to? I found these movies {}.".format( ','.join([self.titles[i][0] for i in title_list]))
        elif (len(title_list) == 0):
            response = "HONK I havent heard of {} before.".format(titles[0])
            possible_title = self.find_movies_closest_to_title(titles[0])
            if len(possible_title) == 0:
                return "HONK TO DO HONK I GOT NO CLUE WHAT YOU ARE TALKING ABOUT"
            return (" HONK I can spell better and I dont even have hands. Perhaps you wanted one of these movies? {} HONK!".format( ','.join([self.titles[i][0] for i in possible_title])))


        if self.creative:
            response = "I processed {} in creative mode!!".format(line)

        else:

            if sentiment == 1:
                # need to implement some sort of caching here.
                response = r.choice(positive_rec).format(titles[0]) +  followup
                self.times += 1
            elif sentiment == -1:
                response = r.choice(negative_rec).format(titles[0]) + followup
                self.times += 1
            else:
                response = r.choice(unknown_rec).format(titles[0])
            self.vec[title_list[0]] = sentiment
        
        
        if self.times >= 5:
            self.loop_rec()

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
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
        title_pat = re.compile('"(.+)"')
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
        if self.levenshtein(movie, title) <= edit_distance:
            return True
        else:
            return title in movie

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
            if self.title_match(title.upper(), movie.upper()):
                r.append(i)
            elif self.creative and self.title_match(title.upper(), movie.upper(), substring_match=True):
                r.append(i)
        
        return r

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
        preprocessed_input = preprocessed_input.lower().replace("'", "").split()
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

        input_sentiment = 0
        for i in range(len(preprocessed_input)):
            delta = 0
            word_sentiment = self.sentiment.get(preprocessed_input[i], '') # default to empty string
            if word_sentiment == 'pos':
                delta = 1
            elif word_sentiment == 'neg':
                delta = -1
            if i > 0 and NEGATION_RE.search(preprocessed_input[i-1]):
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
        return [('', 0)]

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
        for i in range(len(self.titles)):
            movie = self.titles[i][0]
            if self.title_match(title.upper(), movie.upper(), edit_distance=max_distance):
                r.append(i)
        return r

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
            return False
        # Filter the candiates out if they don't match the clarification (substring / year match)
        filtered_candidates = list(filter(remains_valid, candidates))

        try:
            #If the clarification is an int, it might be an index into our list (1-indexed)
            idx = int(clarification)
            if 0 < idx <= len(candidates) and candidates[idx-1] not in filtered_candidates:
                filtered_candidates.append(candidates[idx-1])
        except ValueError:
            pass

        # check against time requests
        if 'recent' in clarification or 'newest' in clarification:
            # This function will try to find the year the movie came out in
            def get_year(title):
                try:
                    return int(year_regex.findall(self.titles[title][0]))
                except (ValueError,TypeError) as e:
                    return -1
            # If we want the "newest" movie, we add the max year (newest) movie to filtered_candidates
            filtered_candidates.append(max(candidates, key = lambda t : get_year(t)))
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


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
