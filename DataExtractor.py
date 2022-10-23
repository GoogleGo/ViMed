class DataExtractor:
    def getSymptoms(self):
        symptoms = set()

        # Open Dataset.csv file
        with open('DiseaseSymptomDataset/dataset.csv', 'r') as f:
            # Ignore the first line
            f.readline()
            # Read the rest of the file
            for line in f:
                sub = line.split(',')[1:]
                if '' in sub:
                    sub = sub[:sub.index('')]
                symptoms.update(sub)
        symptoms = list(symptoms)


        for i in range(len(symptoms)):
            # remove leading and trailing whitespace
            symptoms[i] = symptoms[i].strip()
            # Replace all underscores with spaces
            symptoms[i] = symptoms[i].replace('_', ' ')
            # Capitalize the first letter of each word
            symptoms[i] = symptoms[i].title()

        # Remove any blank entries
        symptoms = [x for x in symptoms if x != '']
        unsorted = symptoms.copy()
        symptoms.sort()
        return symptoms, unsorted