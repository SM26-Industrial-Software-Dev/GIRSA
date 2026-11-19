from HebrewTransliterationEngine import HebrewTransliterationEngine


trans = HebrewTransliterationEngine()
text = 'There\'s the <i>Midrash</i> that says that when God offered the Jewish people the <i>Torah</i>, they said they wanted it, and God says, "I need an <i>arev</i>, I need a guarantor that you\'re going to uphold it." So they responded, "Our children will be the guarantors." So that\'s the idea, the whole notion that we will raise children who will follow in this path and that will justify our receiving the <i>Torah</i> in the first place.'
print(trans.process_transcript(text))
