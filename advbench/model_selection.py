import pandas as pd



class ModelSelection:
    def __init__(self, df):
        self.df = df
        
        metric_name = self.df['Metric-Name'].iloc[0]
        self.sort_ascending = False if 'Accuracy' in metric_name else True

        validation_df, test_df = self.select_epoch()

        validation_df['trial_rank'] = validation_df.groupby(
            'trial_seed'
        )['Metric-Value'].rank(method='dense', ascending=self.sort_ascending)
        test_df['trial_rank'] = validation_df['trial_rank'].tolist()

        self.trial_values = []
        for _, df in test_df.groupby('trial_seed'):
            self.trial_values.append(
                df[df.trial_rank == 1.0]['Metric-Value'].iloc[0])

class LastStep(ModelSelection):
    """Model selection from the *last* step of training."""

    NAME = 'LastStep'

    def __init__(self, df):
        super(LastStep, self).__init__(df)

    def select_epoch(self):
        last_step = max(self.df.Epoch.unique())
        self.df = self.df[self.df.Epoch == last_step]

        validation_df = self.df[self.df.Split == 'Validation'].copy()
        test_df = self.df[self.df.Split == 'Test'].copy()

        return validation_df, test_df
        
class EarlyStop(ModelSelection):
    """Model selection from the *best* of training."""

    NAME = 'EarlyStop'

    def __init__(self, df):
        super(EarlyStop, self).__init__(df)

    def select_epoch(self):
        validation_df = self.df[self.df.Split == 'Validation']
        test_df = self.df[self.df.Split == 'Test']

        validation_dfs, test_dfs = [], []
        for (t, s), df in validation_df.groupby(['trial_seed', 'seed']):
            best_epoch = df[df['Metric-Value'] == self.find_best(df)]['Epoch'].iloc[0]

            validation_dfs.append(
                df[df.Epoch == best_epoch])

            test_dfs.append(
                test_df[
                    (test_df.Epoch == best_epoch) &
                    (test_df.seed == s) & 
                    (test_df.trial_seed == t)])
            
        validation_df = pd.concat(validation_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        return validation_df, test_df

    def find_best(self, df):
        if self.sort_ascending is False:
            return df['Metric-Value'].max()
        return df['Metric-Value'].min()
