import pandas as pd


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


def run_test_loops(file_name, n, x_list, y_list, offset, svr_setup, functions):
    df_x = []
    df_y = []
    df_svr = []
    df_func = []
    df_success = []
    df_off_by = []
    df_sd_data = []
    df_sd_prediction = []

    for i_1 in range(len(x_list)):
        for i_2 in range(len(y_list)):
            for i_3 in range(len(svr_setup)):
                for i_4 in range(len(functions)):
                    result_bert = functions[i_4](n, x_list[i_1], y_list[i_2], offset, svr_setup[i_3])

                    df_x.append(x_list[i_1])
                    df_y.append(y_list[i_2])
                    df_svr.append(svr_setup[i_3])
                    df_func.append(functions[i_4])
                    df_success.append(result_bert['Success rate'])
                    df_off_by.append(result_bert['Overall off by'] / n)
                    df_sd_data.append(result_bert['SD data'])
                    df_sd_prediction.append(result_bert['SD predictions'])

    df_dict = {
        'x': df_x,
        'y': df_y,
        'svr': df_svr,
        'func': df_func,
        'success%': df_success,
        'off_by': df_off_by,
        'SD data': df_sd_data,
        'SD prediction': df_sd_prediction
    }

    final_df = pd.DataFrame(df_dict)

    file_name += '.xlsx'
    final_df.to_excel(file_name)
