import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_train.csv")

def count_unique_tags(x):
    return pd.Series({
        'COMMAND_1_COUNT': (x['TAG'] == 'B-COMMAND_1').sum(),
        'COMMAND_2_COUNT': (x['TAG'] == 'B-COMMAND_2').sum(),
        'FOOD_COUNT': (x['TAG'] == 'B-FOOD').sum(),
        'TABLE_COUNT': (x['TAG'] == 'B-TABLE').sum(),
        'QUESTION_COUNT': (x['TAG'] == 'B-QUESTION').sum()
})

sentence_summary = df.groupby('SENTENCE').apply(count_unique_tags).reset_index()

total_command_1_sentences = (sentence_summary['COMMAND_1_COUNT'] > 0).sum()
total_command_2_sentences = (sentence_summary['COMMAND_2_COUNT'] > 0).sum()
print(total_command_1_sentences)
print(total_command_2_sentences)

command_1_food_tags = sentence_summary[sentence_summary['COMMAND_1_COUNT'] > 0]['FOOD_COUNT'].sum()
command_1_table_tags = sentence_summary[sentence_summary['COMMAND_1_COUNT'] > 0]['TABLE_COUNT'].sum()
command_1_question_tags = sentence_summary[sentence_summary['COMMAND_1_COUNT'] > 0]['QUESTION_COUNT'].sum()

command_2_food_tags = sentence_summary[sentence_summary['COMMAND_2_COUNT'] > 0]['FOOD_COUNT'].sum()
command_2_table_tags = sentence_summary[sentence_summary['COMMAND_2_COUNT'] > 0]['TABLE_COUNT'].sum()
command_2_question_tags = sentence_summary[sentence_summary['COMMAND_2_COUNT'] > 0]['QUESTION_COUNT'].sum()

summary_data = {
    'Tag Type': ['COMMAND_1', 'COMMAND_2'],
    'Sentences Total': [total_command_1_sentences, total_command_2_sentences],
    'FOOD Tags Total': [command_1_food_tags, command_2_food_tags],
    'TABLE Tags Total': [command_1_table_tags, command_2_table_tags],
    'QUESTION Tags Total': [command_1_question_tags, command_2_question_tags]
}

summary_df = pd.DataFrame(summary_data)

sentence_sum = summary_df['Sentences Total'].sum()
food_tags_total = summary_df['FOOD Tags Total'].tolist()
table_tags_total = summary_df['TABLE Tags Total'].tolist()
question_tags_total = summary_df['QUESTION Tags Total'].tolist()

tag_types = ['COMMAND_1', 'COMMAND_2']
food_tags_total = [food_tags_total[0], food_tags_total[1]]
table_tags_total = [table_tags_total[0], table_tags_total[1]]
question_tags_total = [question_tags_total[0], question_tags_total[1]]

fig, ax = plt.subplots()

bar_width = 0.25
index = range(len(tag_types))

bar1 = plt.bar(index, food_tags_total, bar_width, label='FOOD Tags Total')
bar2 = plt.bar([i + bar_width for i in index], table_tags_total, bar_width, label='TABLE Tags Total')
bar3 = bar3 = plt.bar([i + 2 * bar_width for i in index], question_tags_total, bar_width, label='QUESTION Tags Total')

plt.xlabel('Tag Type')
plt.ylabel('Count')
plt.title(f'Sentence Total: {sentence_sum}')
plt.xticks([i + bar_width / 2 for i in index], tag_types)

for i, v in enumerate(food_tags_total):
    plt.text(i - 0.05, v + 5, str(v), color='blue', fontweight='bold')

for i, v in enumerate(table_tags_total):
    plt.text(i + bar_width - 0.05, v + 5, str(v), color='orange', fontweight='bold')

for i, v in enumerate(question_tags_total):
    plt.text(i + 2 * bar_width - 0.05, v + 5, str(v), color='green', fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig("data_visualozation_bar_chart.png")
plt.show()

tag_types = ['COMMAND_1', 'COMMAND_2']
labels = [
    f'FOOD Tags of {tag_types[0]}\nTotal:{food_tags_total[0]}',
    f'FOOD Tags of {tag_types[1]}\nTotal:{food_tags_total[1]}',
    f'Table Tags of {tag_types[0]}\nTotal:{table_tags_total[0]}',
    f'Table Tags of {tag_types[1]}\nTotal:{table_tags_total[1]}',
    f'Question Tags of {tag_types[0]}\nTotal:{question_tags_total[0]}',
    f'Question Tags of {tag_types[1]}\nTotal:{question_tags_total[1]}'  
]

# food_tags_total = [food_tags_total[0], food_tags_total[1]]
sizes_command = [food_tags_total[0], food_tags_total[1], table_tags_total[0], table_tags_total[1], question_tags_total[0], question_tags_total[1]]
# table_tags_total = [table_tags_total[0], table_tags_total[1]]

plt.figure(figsize=(7, 7))
plt.pie(sizes_command, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title(f'Sentence Total: {sentence_sum}')
plt.savefig("data_visualozation_pie_chart.png")
plt.show()