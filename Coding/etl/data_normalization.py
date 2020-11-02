import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import os
import logging

def data_norm():

        """ CSV file input, columns and data cleansing and export to new CSV file
    
        Parameters
        ----------
        No parameters

        Returns:
        ----------
        No returns, output a file to hardcoded path
        """
        
        # Working with local directory to get the correct input path
        #os.chdir('..')
        #os.chdir('Datasets/')
        logging.warning(os.getcwd())

        # Reading raw input CSV from Coffee Quality Institute
        df_raw = pd.read_csv('Datasets/Coffee-quality-institute.csv')
        df_raw = df_raw[['ID',
                        'Species', 
                        'Country.of.Origin', 
                        'Lot.Number', 
                        'Altitude', 
                        'unit_of_measurement', 
                        'Number.of.Bags', 
                        'Bag.Weight', 
                        'Aroma', 
                        'Flavor', 
                        'Aftertaste', 
                        'Acidity', 
                        'Body', 
                        'Balance', 
                        'Uniformity', 
                        'Clean.Cup', 
                        'Sweetness', 
                        'Cupper.Points', 
                        'Moisture', 
                        'Category.One.Defects', 
                        'Color'
                        ]]

        logging.warning('CSV File read successfully\n')
        logging.warning('.')

        # Renaming columns according to friendly labels
        df_renamed = df_raw.rename(columns={'Country.of.Origin': 'ORIGIN_COUNTRY',
                                        'Species': 'SPECIES',
                                        'Lot.Number': 'LOT_NUMBER',
                                        'Altitude': 'ALTITUDE',
                                        'unit_of_measurement': 'ALTITUDE_UOM_NN',
                                        'Number.of.Bags': 'BAG_AVAILABILITY',
                                        'Bag.Weight': 'BAG_WEIGHT_NN',
                                        'Aroma': 'AROMA',
                                        'Flavor': 'FLAVOR',
                                        'Aftertaste': 'AFTERTASTE',
                                        'Acidity': 'ACIDITY',
                                        'Body': 'BODY',
                                        'Balance': 'BALANCE',
                                        'Uniformity': 'UNIFORMITY',
                                        'Clean.Cup': 'CLEAN_CUP',
                                        'Sweetness': 'SWEETNESS',
                                        'Cupper.Points': 'CUPPER_POINTS',
                                        'Moisture': 'MOISTURE',
                                        'Category.One.Defects': 'CATEGORY_ONE_DEFECTS',
                                        'Color': 'COLOR'
                                        }
                                )

        # Applying columns definition according to model needs
        df_treated = df_renamed[(df_renamed['ID'].notnull()) &
                                (df_renamed['SPECIES'] == 'Arabica') &
                                (df_renamed['BAG_AVAILABILITY'].notnull()) &
                                (df_renamed['BAG_WEIGHT_NN'].notnull()) &
                                (df_renamed['AROMA'].notnull()) &
                                (df_renamed['FLAVOR'].notnull()) &
                                (df_renamed['AFTERTASTE'].notnull()) &
                                (df_renamed['ACIDITY'].notnull()) &
                                (df_renamed['BODY'].notnull()) &
                                (df_renamed['BALANCE'].notnull()) &
                                (df_renamed['UNIFORMITY'].notnull()) &
                                (df_renamed['CLEAN_CUP'].notnull()) &
                                (df_renamed['SWEETNESS'].notnull()) &
                                (df_renamed['CUPPER_POINTS'].notnull()) &
                                (df_renamed['MOISTURE'].notnull()) &
                                (df_renamed['CATEGORY_ONE_DEFECTS'].notnull()) &
                                (df_renamed['COLOR'].notnull())
                        ]

        df_treated[['BAG_WEIGHT_NN_VALUE', 'BAG_WEIGHT_NN_UNIT']] = df_treated['BAG_WEIGHT_NN'].str.split(' ', expand = True)
        df_treated = df_treated.drop(columns = ['BAG_WEIGHT_NN'])

        # Creating dummy BAG_WEIGHT_KG column to work on .loc next
        df_treated['BAG_WEIGHT_KG'] = np.NaN

        # Normalizing BAG_AVAILABILITY
        # Columns in which the unit is kg, gets the kg value
        df_treated.loc[(df_treated['BAG_WEIGHT_NN_UNIT'] == 'kg'),['BAG_WEIGHT_KG']] = df_treated.loc[(df_treated['BAG_WEIGHT_NN_UNIT'] == 'kg')]['BAG_WEIGHT_NN_VALUE']

        # Columns in which the unit is lbs, gets the kg value times 0.4536 in order to convert it from lbs to kg
        df_treated.loc[(df_treated['BAG_WEIGHT_NN_UNIT'] == 'lbs')]['BAG_WEIGHT_KG'] = df_treated[(df_treated['BAG_WEIGHT_NN_UNIT'] == 'lbs')]['BAG_WEIGHT_NN_VALUE'].astype(int)*0.4535

        # Columns with unit different from kg or lbs gets NaN in order to be removed
        df_treated.loc[(df_treated['BAG_WEIGHT_NN_UNIT'] != 'kg') & (df_treated['BAG_WEIGHT_NN_UNIT'] != 'lbs'), ['BAG_WEIGHT_KG']] = np.NaN

        # After normalizing the bag UoM, creating LOT_AVAILABILITY_KG to get the row availability in quilograms
        df_treated['LOT_AVAILABILITY_KG'] = df_treated['BAG_WEIGHT_KG'].astype(float)*df_treated['BAG_AVAILABILITY'].astype(float)
        df_treated = df_treated.drop(columns = ['BAG_WEIGHT_KG', 'BAG_AVAILABILITY', 'BAG_WEIGHT_NN_VALUE', 'BAG_WEIGHT_NN_UNIT'])

        df_treated = df_treated.astype({'ID': int,
                                        'SPECIES': str,
                                        'ORIGIN_COUNTRY': str,
                                        'LOT_NUMBER': str,
                                        'ALTITUDE': str,
                                        'ALTITUDE_UOM_NN': str,
                                        'AROMA': float,
                                        'FLAVOR': float,
                                        'AFTERTASTE': float,
                                        'ACIDITY': float,
                                        'BODY': float,
                                        'BALANCE': float,
                                        'UNIFORMITY': float,
                                        'CLEAN_CUP': float,
                                        'SWEETNESS': float,
                                        'CUPPER_POINTS': float,
                                        'MOISTURE': float,
                                        'CATEGORY_ONE_DEFECTS': float,
                                        'COLOR': str,
                                        'LOT_AVAILABILITY_KG': float
                                        })

        df_opt = df_treated[(df_treated['LOT_AVAILABILITY_KG'].notnull()) &
                        (df_treated['MOISTURE'] != 0)
                        ].reset_index()

        # Changin current directory to export on correct path
        #os.chdir('C:/Users/BC966HL/OneDrive - EY/4. Mestrado/Dev/Datasets')

        logging.warning(os.getcwd())
        df_opt.to_csv('Datasets/etl/cleaned_cqi_file.csv', sep = ';')

        logging.warning('.')
        logging.warning('Cleaned CSV file exported to path successfully')
