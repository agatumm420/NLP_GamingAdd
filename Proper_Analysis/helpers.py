from sklearn import preprocessing


def regression_results(answers, results, off_set):
    predicted = 0
    failed = 0
    overall_off_by = 0
    for score, prediction in zip(answers, results):
        score = int(score)
        if score - off_set < prediction < score + off_set:
            print(f'PREDICTED')
            predicted += 1
        else:
            print('FAILED')
            failed += 1
        print(f'Prediction: {round(prediction, 2)}, Real score: {score}')
        off_by = score - prediction
        print(f'Off by {off_by}')
        print('\n')
        overall_off_by += abs(off_by)

    success_rate = predicted / (predicted + failed) * 100
    print(f'Predicted: {predicted}')
    print(f'Failed: {failed}')
    print(f'Success rate: {success_rate}')
    print(f'Overall off by: {overall_off_by}')

    efficiency_dict = {
        'Success rate': success_rate,
        'Overall off by': overall_off_by
    }
    return efficiency_dict


def print_categorical_results(answers, results, real_score, thresh_hold):
    predicted = 0
    failed = 0

    print(real_score)
    for prediction, real_classification, real_score in zip(results, answers, real_score):

        if prediction == real_classification:
            predicted += 1
            if real_classification == 'Non problematic':
                continue

            print('PREDICTED')

        else:
            print('FAILED')
            print(f'by {abs(thresh_hold - real_score)} points')
            failed += 1

        print(f'Prediction: {prediction}, True state: {real_classification} | Points: {real_score}\n')

    success_rate = predicted / (predicted + failed) * 100
    print(f'Predicted: {predicted}')
    print(f'Failed: {failed}')
    print(f'Success rate: {success_rate}')


def categorize(arr, thresh_hold):

    class Category:
        non_problematic = 'Non problematic'
        problematic = 'Problematic'

    categorized_list = []
    for item in arr:
        if item < thresh_hold:
            categorized_list.append(Category.non_problematic)
        elif item >= thresh_hold:
            categorized_list.append(Category.problematic)
        else:
            print(f'Error with {item}')
    return categorized_list


def connect_answers(column, *args):
    for i in range(len(column)):
        i = i + 1
        text = ''
        for arg in args:
            if arg[i][-1] != '.':
                text += f'{arg[i]}. '
            else:
                text += f'{arg[i]} '
        column[i] = text


def normalize_dataset(column):
    normalized_column = preprocessing.normalize([column])
    return normalized_column[0]
