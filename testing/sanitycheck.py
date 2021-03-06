#!/usr/bin/env python

# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
#
# Usage:
#   python sanity_check.py --recommender
#   python sanity_check.py --binarize
######################################################################
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from chatbot import Chatbot

import argparse
import numpy as np
import math


def assertNumpyArrayEquals(givenValue, correctValue, failureMessage):
    try:
        assert np.array_equal(givenValue, correctValue)
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def assertListEquals(givenValue, correctValue, failureMessage, orderMatters=True):
    try:
        if orderMatters:
            assert givenValue == correctValue
            return True
        givenValueSet = set(givenValue)
        correctValueSet = set(correctValue)
        assert givenValueSet == correctValueSet
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def assertEquals(givenValue, correctValue, failureMessage):
    try:
        assert givenValue == correctValue
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def test_similarity():
    print("Testing similarity() functionality...")
    chatbot = Chatbot(False)

    x = np.array([1, 1, -1, 0], dtype=float)
    y = np.array([1, 0, 1, -1], dtype=float)

    self_similarity = chatbot.similarity(x, x)
    if not math.isclose(self_similarity, 1.0):
        print('Unexpected cosine similarity between {} and itself'.format(x))
        print('Expected 1.0, calculated {}'.format(self_similarity))
        print()
        return False

    ortho_similarity = chatbot.similarity(x, y)
    if not math.isclose(ortho_similarity, 0.0):
        print('Unexpected cosine similarity between {} and {}'.format(x, y))
        print('Expected 0.0, calculated {}'.format(ortho_similarity))
        print()
        return False

    print('similarity() sanity check passed!')
    print()
    return True


def test_binarize():
    print("Testing binarize() functionality...")
    chatbot = Chatbot(False)
    if assertNumpyArrayEquals(
            chatbot.binarize(np.array([[1, 2.5, 5, 0]])),
            np.array([[-1., -1., 1., 0.]]),
            "Incorrect output for binarize(np.array([[1, 2.5, 5, 0]]))."
    ):
        print("binarize() sanity check passed!")
    print()


def test_extract_titles():
    print("Testing extract_titles() functionality...")
    chatbot = Chatbot(False)
    if assertListEquals(
            chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook"')),
            ["The Notebook"],
            "Incorrect output for extract_titles(chatbot.preprocess(\'I liked \"The Notebook\"\'))."
    ) and assertListEquals(
        chatbot.extract_titles(chatbot.preprocess('No movies here!')),
        [],
        "Incorrect output for extract_titles(chatbot.preprocess('No movies here!')).",
    ):
        print('extract_titles() sanity check passed!')
    print()


def test_find_movies_by_title():
    print("Testing find_movies_by_title() functionality...")
    chatbot = Chatbot(False)
    if assertListEquals(
            chatbot.find_movies_by_title("The American President"),
            [10],
            "Incorrect output for find_movies_by_title('The American President')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic"),
        [1359, 2716],
        "Incorrect output for find_movies_by_title('Titanic').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic (1997)"),
        [1359],
        "Incorrect output for find_movies_by_title('Titanic (1997)').",
    ):
        print('find_movies_by_title() sanity check passed!')
    print()


def test_extract_sentiment():
    print("Testing extract_sentiment() functionality...")
    chatbot = Chatbot(False)
    if assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I like \"Titanic (1997)\".")),
            1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I like \"Titanic (1997)\".\'))"
    ) and assertEquals(
        chatbot.extract_sentiment(chatbot.preprocess("I loved \"Zootopia\"")),
        2,
        "Incorrect output for extract_sentiment(chatbot.preprocess(\'I loved \"Zootopia\"\'))"
    ) and assertEquals(
        chatbot.extract_sentiment(chatbot.preprocess("\"Zootopia\" was terrible.")),
        -2,
        "Incorrect output for extract_sentiment(chatbot.preprocess(\'\"Zootopia\" was terrible.\'))"
    ) and assertEquals(
        chatbot.extract_sentiment(chatbot.preprocess("I really reeally liked \"Zootopia\"!!!")),
        2,
        "Incorrect output for extract_sentiment(chatbot.preprocess(\'I really reeally liked \"Zootopia\"!!!\'))"
    ) and assertEquals(
        chatbot.extract_sentiment(chatbot.preprocess("I saw \"Titanic (1997)\".")),
        0,
        "Incorrect output for extract_sentiment(chatbot.preprocess(\'I saw  \"Titanic (1997)\".\'))"
    ) and assertEquals(
        chatbot.extract_sentiment(chatbot.preprocess("I didn't enjoy \"Titanic (1997)\".")),
        -1,
        "Incorrect output for extract_sentiment(chatbot.preprocess(\'I didn't enjoy \"Titanic (1997)\"\'.))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I loved \"10 Things I Hate About You\".")),
            2,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I loved \"10 Things I Hate About You\"\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I liked \"She's the Man\".")),
            1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I liked \"She's the Man\"\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I liked \"Sydney White\".")),
            1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I liked \"Sydney White\"\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I liked \"Easy A\".")),
            1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I liked \"Easy A\"\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("\"Easy A\" is a movie I hate.")),
            -2,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'\"Easy A\" is a movie I hate.\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I hated \"Easy A\".")),
            -2,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I hated \"Easy A\".\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("This movie, \"Easy A\", is really bad.")),
            -2,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'This movie, \"Easy A\", is really bad.\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("\"Titanic (1997)\" started out terrible, but the ending was totally great and I loved it!")),
            2,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'\"Titanic (1997)\" started out terrible, but the ending was totally great and I loved it!\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I didn't really like \"Titanic (1997)\".")),
            -1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I didn't really like \"Titanic (1997)\".\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I never liked \"Titanic (1997)\".")),
            -1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I never liked \"Titanic (1997)\".\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I really enjoyed \"Titanic (1997)\".")),
            1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I really enjoyed \"Titanic (1997)\".\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I don't think I really enjoyed \"Titanic (1997)\".")),
            -1,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I don't think I really enjoyed \"Titanic (1997)\".\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I don't think I enjoyed \"Titanic (1997)\" because the plot was really bad and the acting was terrible.")),
            -2,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I don't think I enjoyed \"Titanic (1997)\" because the plot was really bad and the acting was terrible.\'))"
    ) and assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess("I didn't like \"Titanic (1997)\" at the beginning, but at the end I thought it was great and really amazing.")),
            2,
            "Incorrect output for extract_sentiment(chatbot.preprocess(\'I didn't like \"Titanic (1997)\" at the beginning, but at the end I thought it was great and really amazing.\'))"
    ):
        print('extract_sentiment() sanity check passed!')
    print()


def test_extract_sentiment_for_movies():
    print("Testing test_extract_sentiment_for_movies() functionality...")
    chatbot = Chatbot(True)
    if assertListEquals(
            chatbot.extract_sentiment_for_movies(chatbot.preprocess("I liked both \"I, Robot\" and \"Ex Machina\".")),
            [("I, Robot", 1), ("Ex Machina", 1)],
            "Incorrect output for test_extract_sentiment_for_movies("
            "chatbot.preprocess(\"I liked both \"I, Robot\" and \"Ex Machina\".))\"",
            orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies(chatbot.preprocess("I liked \"I, Robot\" but not \"Ex Machina\".")),
        [("I, Robot", 1), ("Ex Machina", -1)],
        "Incorrect output for test_extract_sentiment_for_movies("
        "chatbot.preprocess(\"I liked \"I, Robot\" but not \"Ex Machina\".))\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies(chatbot.preprocess("I liked \"I, Robot\", \"Ex Machina\", and \"Braveheart\".")),
        [("I, Robot", 1), ("Ex Machina", 1), ("Braveheart", 1)],
        "Incorrect output for test_extract_sentiment_for_movies("
        "chatbot.preprocess(\"I liked \"I, Robot\", \"Ex Machina\", and \"Braveheart\".))\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies(chatbot.preprocess("I hated \"I, Robot\", \"Ex Machina\", and \"Braveheart\".")),
        [("I, Robot", -2), ("Ex Machina", -2), ("Braveheart", -2)],
        "Incorrect output for test_extract_sentiment_for_movies("
        "chatbot.preprocess(\"I hated \"I, Robot\", \"Ex Machina\", and \"Braveheart\".))\"",
        orderMatters=False
    ):
        print('extract_sentiment_for_movies() sanity check passed!')
    print()


def test_find_movies_closest_to_title():
    print("Testing find_movies_closest_to_title() functionality...")
    chatbot = Chatbot(True)

    misspelled = "Sleeping Beaty"

    if assertListEquals(
            chatbot.find_movies_closest_to_title(misspelled, max_distance=3),
            [1656],
            "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format(misspelled, 3),
            orderMatters=False
    ) and assertListEquals(
            chatbot.find_movies_closest_to_title("Te", max_distance=3),
            [8082, 4511, 1664],
            "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("Te", 3),
            orderMatters=False
    ) and assertListEquals(
            chatbot.find_movies_closest_to_title("BAT-MAAAN", max_distance=3),
            [524, 5743],
            "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("BAT-MAAAN", 3),
            orderMatters=False
    ) and assertListEquals(
            chatbot.find_movies_closest_to_title("Blargdeblargh", max_distance = 4),
            [],
            "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format("Blargdeblargh", 4),
            orderMatters=False
    ):
        print('find_movies_closest_to_title() sanity check passed!')
    print()
    return True


def test_disambiguate():
    print("Testing disambiguate() functionality...")
    chatbot = Chatbot(True)
    if assertListEquals(
            chatbot.disambiguate("1997", [1359, 2716]),
            [1359],
            "Incorrect output for disambiguate('{}', {})".format("1997", [1359, 2716]),
            orderMatters=False
    ) and assertListEquals(
            chatbot.disambiguate("2", [1142, 1357, 2629, 546]),
            [1357],
            "Incorrect output for disambiguate('{}', {})".format("2", [1142, 1357, 2629, 546]),
            orderMatters=False
    ) and assertListEquals(
            chatbot.disambiguate("Sorcerer's Stone", [3812, 1267]),
            [3812],
            "Incorrect output for disambiguate('{}', {})".format("Sorcerer's Stone", [3812, 1267]),
            orderMatters=False
    ) and assertListEquals(
            chatbot.disambiguate("2", [8082, 4511, 1664]),
            [4511],
            "Incorrect output for disambiguate('{}', {})".format("2", [8082, 4511, 1664]),
            orderMatters=False
    ) and assertListEquals(
            chatbot.disambiguate("most recent", [524, 5743]),
            [524],
            "Incorrect output for disambiguate('{}', {})".format("most recent", [524, 5743]),
            orderMatters=False
    ) and assertListEquals(
            chatbot.disambiguate("the Goblet of Fire one", [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842]),
            [6294],
            "Incorrect output for disambiguate('{}', {})".format("the Goblet of Fire one", [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842]),
            orderMatters=False
    ) and assertListEquals(
            chatbot.disambiguate("the second one", [3812, 6294, 4325, 5399, 6735, 7274, 7670, 7842]),
            [6294],
            "Incorrect output for disambiguate('{}', {})".format("the second one", [3812, 6294, 4325, 5399, 6735, 7274, 7670, 7842]),
            orderMatters=False
    ):
        print('disambiguate() sanity check passed!')
    print()
    return True


def test_recommend():
    print("Testing recommend() functionality...")
    chatbot = Chatbot(False)

    user_ratings = np.array([1, -1, 0, 0, 0, 0])
    all_ratings = np.array([
        [1, 1, 1, 0],
        [1, -1, 0, -1],
        [1, 1, 1, 0],
        [0, 1, 1, -1],
        [0, -1, 1, -1],
        [-1, -1, -1, 0],
    ])
    recommendations = chatbot.recommend(user_ratings, all_ratings, 2)

    if assertListEquals(recommendations, [2, 3], "Recommender test failed"):
        print("recommend() sanity check passed!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Sanity checks the chatbot. If no arguments are passed, all checks are run; you can use the '
                    'arguments below to test specific parts of the functionality.')

    parser.add_argument('-b', '--creative', help='Tests all of the creative function', action='store_true')

    args = parser.parse_args()
    testing_creative = args.creative

    test_extract_sentiment() ##deleteee

    test_extract_titles()
    test_find_movies_by_title()
    test_extract_sentiment()
    test_recommend()
    test_binarize()
    test_similarity()

    if testing_creative:
        test_find_movies_closest_to_title()
        test_extract_sentiment_for_movies()
        test_disambiguate()


if __name__ == '__main__':
    main()
