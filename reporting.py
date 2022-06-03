import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import diagnostics
from sklearn import metrics
import logging

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle
from reportlab.lib.colors import lavender, red, green

from config import DATA_PATH, OUTPUT_MODEL_PATH, TEST_DATA_PATH


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

###############Load config.json and get path variables


##############Function for reporting
def plot_confusion_matrix():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    logger.info("Predicting data")
    _, _, y = diagnostics.load_data()
    y_pred = diagnostics.model_predictions()

    # logger.info(f"y_pred: {y_pred}")
    cm = metrics.confusion_matrix(y, y_pred)

    _ = sns.heatmap(cm)
    plt.title(f'Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # write the confusion matrix to the workspace
    fig = os.path.join(OUTPUT_MODEL_PATH, 'confusionmatrix.png')
    plt.savefig(fig)
    
    return


def _get_statistics():
    """
    Get data statistics and missing percentage of each column
    in pandas dataframe to draw table in the PDF report
    Returns:
        pd.DataFrame: Train data summary
    """
    stats = diagnostics.dataframe_summary()
    missing = diagnostics.missing_data()

    data = {'Column Name': [k for k in missing.keys()]}
    data['Missing %'] = [missing[column]['percentage']
                         for column in data['Column Name']]

    temp_col = list(stats.keys())[0]
    for stat in stats[temp_col].keys():
        data[stat] = [round(stats[column][stat],2)
                      if stats.get(column,None)
                      else '-' for column in data['Column Name']]

    return data


def generate_pdf_report():
    """
    Generate PDF report that includes ingested data information, model scores
    on test data and diagnostics of execution times and packages
    """
    pdf = canvas.Canvas(os.path.join(OUTPUT_MODEL_PATH,
                        'summary_report.pdf'),
                        pagesize=A4)

    pdf.setTitle("Model Summary Report")

    pdf.setFontSize(24)
    pdf.setFillColorRGB(31 / 256, 56 / 256, 100 / 256)
    pdf.drawCentredString(300, 800, "Model Summary Report")

    # Ingest data section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 750, "Ingested Data")

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 725, "List of files used:")

    # Ingested files
    with open(os.path.join(DATA_PATH, "ingestedfiles.txt")) as file:
        pdf.setFontSize(12)
        text = pdf.beginText(40, 705)
        text.setFillColor('black')

        for line in file.readlines():
            text.textLine(line.strip('\n'))
            print(line)

        pdf.drawText(text)

    # Data statistics and missing percentage
    data = _get_statistics()
    data_df = pd.DataFrame(data)
    data_table = data_df.values.tolist()
    data_table.insert(0, list(data_df.columns))

    # Draw summary table
    stats_table = Table(data_table)
    stats_table.setStyle([
        ('GRID', (0, 0), (-1, -1), 1, 'black'),
        ('BACKGROUND', (0, 0), (-1, 0), lavender)
    ])

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 645, "Statistics Summary")

    stats_table.wrapOn(pdf, 40, 520)
    stats_table.drawOn(pdf, 40, 520)

    # Trained model section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 490, "Trained Model Scoring on Test Data")

    pdf.setFontSize(12)
    pdf.setFillColorRGB(128 / 256, 128 / 256, 128 / 256)
    pdf.drawString(25, 480, "testdata.csv")

    # Model score
    with open(os.path.join(OUTPUT_MODEL_PATH, "latestscore.txt")) as file:
        pdf.setFontSize(12)
        pdf.setFillColor('black')
        pdf.drawString(40, 460, file.read())

    # Model confusion matrix
    pdf.drawInlineImage(
        os.path.join(
            OUTPUT_MODEL_PATH,
            'confusionmatrix.png'),
        40,
        150,
        width=300,
        height=300)

    # New page
    pdf.showPage()

    # Diagnostics section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 780, "Diagnostics")

    # Execution time
    timings = diagnostics.execution_time()

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 755, "Execution times:")

    pdf.setFontSize(12)
    text = pdf.beginText(40, 735)
    text.setFillColor('black')

    for time in timings:
        for k, v in time.items():
            text.textLine(f"{k} = {round(v, 4)}")

    pdf.drawText(text)

     # Draw outdated dependencies table
    data = diagnostics.outdated_packages_list()

    table_style = TableStyle()
    table_style.add('GRID', (0, 0), (-1, -1), 1, 'black')
    table_style.add('BACKGROUND', (0, 0), (-1, 0), lavender)

    for row, values in enumerate(data[1:], start=1):
        if(values[1] != values[2]):
            table_style.add('TEXTCOLOR', (1, row), (1, row), red)
            table_style.add('TEXTCOLOR', (2, row), (2, row), green)

    depend_table = Table(data)
    depend_table.setStyle(table_style)

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 690, "Outdated Dependencies")

    pdf.setFontSize(12)
    pdf.setFillColorRGB(128 / 256, 128 / 256, 128 / 256)
    pdf.drawString(35, 675, "Red = unavailable/outdated/out of version specifier")
    pdf.drawString(35, 665, "Green = updatable")

    depend_table.wrapOn(pdf, 40, 235)
    depend_table.drawOn(pdf, 40, 235)

    pdf.save()


if __name__ == '__main__':
    logger.info("Generating confusion matrix")
    plot_confusion_matrix()
    sys.exit()
    logger.info("Generating PDF report")
    generate_pdf_report()
