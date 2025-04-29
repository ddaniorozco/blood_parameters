# %% Importing libraries
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import best_segment.wide.utils.cross_section_utils as csu
import best_segment.wide.utils.segment_utils as segu
import best_segment.wide.utils.skeleton_utils as su
from best_segment.wide.utils.masking_utils import Masker
from best_segment.wide.utils.sharpness_utils import SharpnessEstimator
import xgboost as xgb


# %% Feature extractor class - training
class CropFeatureExtractor:
    """
    Extracts features from a crop image.
    
    Args:
        image_path (str): The path to the image file.
        segment_line (object): The segment line object.
        depth (float): The depth value.
        exposure_time (float): The exposure time value.
        **kwargs: Additional keyword arguments for the Masker object.
    """
    
    def __init__(self, data_paths, sharpness_th=0, segment_size=20, **kwargs):
        if isinstance(data_paths, str):
            self.data_paths = [data_paths]
        else:
            self.data_paths = data_paths
        self.image_roots = [os.path.join(data_path, "test", "images") for data_path in self.data_paths]
        self.sample_data_files = [os.path.join("/home", "omri", "Datasets", os.path.basename(data_path), "df_line_wide_extrapolated", "df_final.csv") for data_path in self.data_paths]

        self.masker = Masker(**kwargs)
        self.sharpness_estimator = SharpnessEstimator()
        self.cross_sectioner = csu.CrossSection()
        self.sharpness_th = sharpness_th
        self.segment_size = segment_size
        self.rfc = None
        self._features = []
        self.features_pca = None
        self.features_tsne = None
        self.columns_to_ignore = ['name', 'velocity', 'image_root', 'data_file', 'index_in_file']
    
    def mask_crop(self, crop, line_points=None):
        """
        Masks the given crop image.
        
        Args:
            crop (ndarray): The crop image.
        
        Returns:
            ndarray: The masked crop image.
        """
        return self.masker.produce_mask(crop, line_points=line_points)
      
    def estimate_sharpness(self, crop, mask=None):
        """
        Estimates the sharpness of the given crop image.
        
        Args:
            crop (ndarray): The crop image.
            mask (ndarray, optional): The mask image. Defaults to None.
        
        Returns:
            float: The sharpness value.
        """
        if mask is None:
            mask = self.mask_crop(crop)
        return self.sharpness_estimator.estimate_sharpness(crop, mask)
    
    def get_features(self):
        """
        Extracts features from the image.
        """
        for image_root, samples_data_file in zip(self.image_roots, self.sample_data_files):
            sample_data = pd.read_csv(samples_data_file)
            for idx, sample in sample_data.iterrows():
                features = {"image_root": image_root, "data_file": samples_data_file, 
                            "index_in_file": idx, "name": sample['wide_image_name'], 
                            "classification": "bad" if sample['classification'] == 0 else "good", 
                            "classification_probability": sample['classification_probability'],
                            "depth": sample['z_position'] - sample['glass_position'], 
                            "exposure_time": sample['exposure_time'],
                            "analysis_status": "not_analyzed"}
                image_path = os.path.join(image_root, sample['wide_image_name'])
                crop = np.array(Image.open(image_path+".png").convert('L'))
                line = segu.SampleLine(sample)
                line_segment = line.get_line()
                mask = self.mask_crop(crop, line_segment)
                if mask is not None:
                    sharpness = self.estimate_sharpness(crop, mask)
                    if sharpness >= self.sharpness_th:
                        features.update({"sharpness": sharpness})
                        mask = np.array(mask) if not isinstance(mask, np.ndarray) else mask
                        skeleton = self.masker.produce_skeleton(crop, mask, line_segment)
                        if skeleton is not None:
                            spline_x, spline_y, ts, segment_size = su.sample_skeleton(skeleton, self.segment_size)
                            sampled_skeleton = np.array([spline_y(ts), spline_x(ts)]).T
                            distance_to_line_center_ind = np.argmin(np.linalg.norm(sampled_skeleton - line.get_line_center(), axis=-1), axis=0)
                            t = np.linspace(ts[distance_to_line_center_ind]-segment_size/2, ts[distance_to_line_center_ind]+segment_size/2, 100)
                            skeleton_segment = np.array([spline_y(t), spline_x(t)]).T
                            _, closest_inds,  min_distance = segu.intersect(skeleton_segment, line_segment)
                            if min_distance <= 10:
                                ts0 = t[closest_inds[0]]
                                t = np.linspace(ts0-segment_size/2, ts0+segment_size/2, 10)
                                ts0_plus = ts0 + segment_size/len(t)
                                ts0_minus = ts0 - segment_size/len(t) 
                               
                                # Vectors from the intersection points
                                line_vector = line_segment[1] - line_segment[0]
                                skeleton_vector = np.array([
                                    spline_y(ts0_plus) - spline_y(ts0_minus), spline_x(ts0_plus) - spline_x(ts0_minus)]).T

                                # Normalize the vectors
                                line_vector = line_vector / np.linalg.norm(line_vector)
                                skeleton_vector = skeleton_vector / np.linalg.norm(skeleton_vector)

                                # Calculate the angle in radians and then convert to degrees
                                angle_rad = np.arccos(np.clip(np.dot(line_vector, skeleton_vector), -1.0, 1.0))
                                angle_deg = np.degrees(angle_rad)

                                features.update({"intersection_angle": angle_deg})
                                curvatures = su.calculate_skeleton_curvature((spline_x, spline_y, t))
                                features.update({"curvature": np.mean(curvatures)}) 
                                normals = su.calculate_skeleton_normals((spline_x, spline_y, t))
                                self.cross_sectioner.set_normal_cross_section_points(np.array([spline_y(t), spline_x(t)]).T, normals)
                                _, fit_params = self.cross_sectioner.fit(crop, return_dict=True, average=True)
                                capi_features = self.cross_sectioner.get_capillary_features(crop, mask, return_dict=True, average=True)
                                if fit_params['fit_amp'] is not None and not np.isnan(fit_params['fit_amp']) and fit_params['fit_width'] < 30 and not np.isnan(capi_features['mask_width']):
                                    features.update(fit_params)
                                    features.update(capi_features)
                                    features.update({"analysis_status": "success"})
                                    print("Feature extracted for sample {}/{}".format(idx+1, len(sample_data)))
                                else:
                                    features.update({"analysis_status": "no_fit"})
                                    print("No fit for sample {}/{}".format(idx+1, len(sample_data)))
                            else:
                                features.update({"analysis_status": "no_intersection"})
                                print("No intersection in sample {}/{}".format(idx+1, len(sample_data)))
                        else:
                            features.update({"analysis_status": "no_skeleton"})
                            print("No skeleton for sample {}/{}".format(idx+1, len(sample_data)))
                    else:
                        features.update({"analysis_status": "low_sharpness"})
                        print("Sharpness below threshold for sample {}/{}".format(idx+1, len(sample_data)))
                else:
                    features.update({"analysis_status": "no_mask"})
                    print("No mask for sample {}/{}".format(idx+1, len(sample_data)))
                self.features.append(features)
        # Convert self.features to a pandas DataFrame
        for idx, feature in enumerate(self.features):
            if feature['analysis_status'] == 'success':
                first_success_feature = idx
                break

        features_dict = {}
        for key in self.features[first_success_feature].keys():
            features_dict[key] = [d[key] if key in d else np.nan for d in self.features]
        self.features = pd.DataFrame(features_dict)
        return self.features
    
    def plot_sample(self, index=0):
        # Load sample
        sample = self._features.iloc[index]
        image_path = os.path.join(sample['image_root'], sample['name'])
        crop = np.array(Image.open(image_path+".png").convert('L'))
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        # Plot the image
        axs[0].imshow(crop, cmap='gray')
        axs[0].set_title("Sample {}: {}, {}".format(index + 1, sample['classification'], sample['analysis_status']))
        # Plot the mask
        mask = self.mask_crop(crop)
        axs[0].contour(mask, levels=[0.5], colors='red')
        # Plot the line
        data_file = sample['data_file']
        sample_data = pd.read_csv(data_file)
        line = segu.SampleLine(sample_data.iloc[sample['index_in_file']])
        line_ends = line.get_whole_line_ends()
        axs[0].plot(line_ends[:, 1], line_ends[:, 0], '--r', linewidth=0.5)
        axs[0].scatter(line_ends[0, 1], line_ends[0, 0], c='r')
        line_ends = line.get_line_ends()
        axs[0].plot(line_ends[:, 1], line_ends[:, 0], 'r')
        # Plot the skeleton
        skeleton = self.masker.produce_skeleton(crop, mask, line.get_line())
        if skeleton is not None:
            axs[0].plot(skeleton[:, 1], skeleton[:, 0], 'b')
            spline_x, spline_y, ts, segment_size = su.sample_skeleton(skeleton, self.segment_size)
            sampled_skeleton = np.array([spline_y(ts), spline_x(ts)]).T
            distance_to_line_center_ind = np.argmin(np.linalg.norm(sampled_skeleton - line.get_line_center(), axis=-1), axis=0)
            t = np.linspace(ts[distance_to_line_center_ind]-segment_size/2, ts[distance_to_line_center_ind]+segment_size/2, 100)
            skeleton_segment = np.array([spline_y(t), spline_x(t)]).T
            line_segment = line.get_line()
            _, closest_inds,  min_distance = segu.intersect(skeleton_segment, line_segment)
            if min_distance <= 10:
                ts0 = t[closest_inds[0]]
                t = np.linspace(ts0-segment_size/2, ts0+segment_size/2, 10)
                axs[0].plot(spline_x(t), spline_y(t), 'g')
        # Dont show axs[1]
        axs[1].axis('off')
        plt.show()

    @property
    def features(self):
        if isinstance(self._features, pd.DataFrame):
            features = self._features[self._features['analysis_status'] == 'success']
            good_features = features.columns.difference(['analysis_status'])
            return features[good_features]
        else:
            return self._features
    
    @features.setter
    def features(self, value):
        self._features = value

    def features_standardize(self, remove_classification=False, scale=True, columns_to_ignore=None):
        """
        Standardizes the features.
        
        Args:
            remove_classification (bool): Whether to remove the 'classification' column. Defaults to False.
            columns_to_standardize (list): The columns to standardize. Defaults to None.
        """
        if columns_to_ignore is None:
            columns_to_standardize = self.features.columns.difference(self.columns_to_ignore)
        else:
            columns_to_standardize = self.features.columns.difference(self.columns_to_ignore + columns_to_ignore)
        if remove_classification:
            columns_to_standardize = columns_to_standardize.difference(['classification'])
        features = self.features[columns_to_standardize]
        if scale:
            return StandardScaler().fit_transform(features)
        else:
            return features

    
    def save_features(self, file_path=None):
        """
        Saves the features to a CSV file.
        
        Args:
            file_path (str): The path to the file.
        """
        if file_path is None:
            file_path = self.sample_data_files[0].replace("df_final.csv", "features.csv")
        self.features.to_csv(file_path, index=False)

    def load_features(self, file_path=None):
        """
        Loads the features from a CSV file.
        
        Args:
            file_path (str): The path to the file.
        """
        if file_path is None:
            file_path = self.sample_data_files[0].replace("df_final.csv", "features.csv")
        self.features = pd.read_csv(file_path)
    
    def fit_rfc(self, scale=False, test_size=0.2, columns_to_ignroe=None, **kwargs):
        if len(self.features) == 0:
            self.get_features()
        if columns_to_ignroe is None:
            columns_to_ignroe = self.columns_to_ignore
        else:
            columns_to_ignroe = columns_to_ignroe + self.columns_to_ignore
        X = self.features_standardize(remove_classification=True, scale=False, columns_to_ignore=columns_to_ignroe)
        y = self.features['classification']
        
        le = LabelEncoder()
        le.fit(['bad', 'good'])
        y = le.transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        rfc = RFC(**kwargs)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', pos_label=le.transform(['good'])[0])
        recall = recall_score(y_test, y_pred, average='binary', pos_label=le.transform(['good'])[0])
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=le.transform(['good'])[0])

        # Print classifier name
        print("~~~~~~~~~ RFC results ~~~~~~~~~")
        # Print metrics
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Print confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Assuming rfc is your trained RandomForestClassifier
        importances = rfc.feature_importances_

        # Print the feature importance
        for feature, importance in zip(self.features.columns.difference(columns_to_ignroe + ["classification"]) , importances):
            print(f"Feature: {feature}, Importance: {importance}")
        self.rfc = rfc
        return rfc
    
    def save_rfc(self, file_path=None, file_name="rfc_model.pkl"):
        """
        Saves the trained RFC model to a file.
        
        Args:
            file_path (str): The path to the file.
        """
        if file_path is None:
            file_path = self.sample_data_files[0].replace("df_final.csv", file_name)
        if self.rfc is not None:
            joblib.dump(self.rfc, file_path)
    
    def load_rfc(self, file_path=None, file_name="rfc_model.pkl"):
        """
        Loads the trained RFC model from a file.
        
        Args:
            file_path (str): The path to the file.
        """
        if file_path is None:
            file_path = self.sample_data_files[0].replace("df_final.csv", file_name)
        self.rfc = joblib.load(file_path)

    def fit_xgb(self, scale=False, test_size=0.2, columns_to_ignore=None, **kwargs):
        if len(self.features) == 0:
            self.get_features()
        if columns_to_ignore is None:
            columns_to_ignore = self.columns_to_ignore
        else:
            columns_to_ignore = columns_to_ignore + self.columns_to_ignore
        X = self.features_standardize(remove_classification=True, scale=False, columns_to_ignore=columns_to_ignore)
        y = self.features['classification']
        
        le = LabelEncoder()
        le.fit(['bad', 'good'])
        y = le.transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        xgb_model = xgb.XGBClassifier(**kwargs)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', pos_label=le.transform(['good'])[0])
        recall = recall_score(y_test, y_pred, average='binary', pos_label=le.transform(['good'])[0])
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=le.transform(['good'])[0])

        # Print classifier name
        print("~~~~~~~~~ XGBoost results ~~~~~~~~~")
        # Print metrics
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Print confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Assuming xgb_model is your trained XGBoost model
        importances = xgb_model.feature_importances_

        # Print the feature importance
        for feature, importance in zip(self.features.columns.difference(columns_to_ignore + ["classification"]) , importances):
            print(f"Feature: {feature}, Importance: {importance}")
        self.xgb_model = xgb_model
        return xgb_model
    
    def save_xgb(self, file_path=None, file_name="xgb_model.pkl"):
        """
        Saves the trained XGBoost model to a file.
        
        Args:
            file_path (str): The path to the file.
        """
        if file_path is None:
            file_path = self.sample_data_files[0].replace("df_final.csv", file_name)
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, file_path)
    
    def load_xgb(self, file_path=None, file_name="xgb_model.pkl"):
        """
        Loads the trained XGBoost model from a file.
        
        Args:
            file_path (str): The path to the file.
        """
        if file_path is None:
            file_path = self.sample_data_files[0].replace("df_final.csv", file_name)
        self.xgb_model = joblib.load(file_path)
    
    def plot_correlations(self):
        """
        Plots the correlation pairplot for the features.
        """
        # Load the data
        data = self.features
        columns_to_standardize = data.columns.difference(['velocity', 'image_root', 'data_file', 'index_in_file'])
        # Clean and convert numerical columns, excluding 'name' and 'classification'
        for column in data.columns:
            if column not in ['name', 'classification']:
                # Remove brackets and convert to numeric, coercing errors to NaN
                data[column] = data[column].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
                data[column] = pd.to_numeric(data[column], errors='coerce')
                data.loc[data['classification'] == 'not_sure', 'classification'] = 'bad'

        # Use Plotly Express to create a scatter matrix
        fig = px.scatter_matrix(data[columns_to_standardize],
                                dimensions=[col for col in data[columns_to_standardize].columns if col not in ['name', 'classification']],
                                color="classification",  # Color code by the 'classification' column
                                title="Scatter matrix of Features vs classification",
                                width=3000,
                                height=3000,
                                color_discrete_map={'good': 'blue', 'bad': 'red', 'not_sure': 'green'})

        fig.show()

    def reduce_dimensions(self, method='both', plot=True, **kwargs):
        """
        Reduces the dimensions of the features.
        
        Args:
            method (str): The method to use. Can be 'pca', 'tsne', or 'both'. Defaults to 'both'.
            **kwargs: Additional keyword arguments for the PCA and TSNE objects.
        """
        data = self.features_standardize(remove_classification=True, scale=True)
        if method == 'pca' or method == 'both':
            pca = PCA(**kwargs)
            self.features_pca = pca.fit_transform(data)
        if method == 'tsne' or method == 'both':
            tsne = TSNE(**kwargs)
            self.features_tsne = tsne.fit_transform(data)
        if plot:
            # Create subplots with titles
            fig = make_subplots(rows=1, cols=2, subplot_titles=('PCA', 't-SNE'))

            # Add scatter plot for PCA results to the first subplot
            if method == 'pca' or method == 'both':
                fig.add_trace(
                    go.Scatter(
                        x=self.features_pca[:, 0],
                        y=self.features_pca[:, 1],
                        mode='markers',
                        marker=dict(
                            color=self.features['classification'].map({'good': 'blue', 'bad': 'red', 'not_sure': 'green'}),
                            size=8),
                        text=self.features.index,  # Add path values to hover text
                        hoverinfo='text',
                        name='PCA'),
                    row=1, col=1)

            # Add scatter plot for t-SNE results to the second subplot
            if method == 'tsne' or method == 'both':
                fig.add_trace(
                    go.Scatter(
                        x=self.features_tsne[:, 0],
                        y=self.features_tsne[:, 1],
                        mode='markers',
                        marker=dict(
                            color=self.features['classification'].map({'good': 'blue', 'bad': 'red', 'not_sure': 'green'}),
                            size=8),
                        text=self.features.index,  # Add path values to hover text
                        hoverinfo='text',
                        name='t-SNE'),
                    row=1, col=2)

            # Update layout to disable the legend on the right
            fig.update_layout(
                showlegend=False)
            # Show the plot
            fig.show()

#%% Testing
if __name__ == "__main__":
    dataset_paths = ["/home/datasets/examples/wide_and_line_frames/segment1_2024_09_17"]
    sharpness_th=0 
    segment_size=20
    line_analyzer = CropFeatureExtractor(dataset_paths, sharpness_th=sharpness_th, segment_size=segment_size)
    features = line_analyzer.get_features()
    #%%
    line_analyzer.plot_sample(76)
    #%%
    columns_to_ignore = ['classification_probability', 'fit_amp', 'fit_width', 'fit_offset', 'fit_fwhm', 'fit_exponent', 'fit_center', 'mask_width', 'exposure_time', 'capillary_background_min_max', 'capillary_std']
    rfc = line_analyzer.fit_rfc(columns_to_ignroe=columns_to_ignore, scale=False, test_size=0.5)
    #%%
    columns_to_ignore = ['classification_probability', 'fit_amp', 'fit_width', 'fit_offset', 'fit_fwhm', 'fit_exponent', 'fit_center', 'mask_width', 'exposure_time', 'capillary_background_min_max', 'capillary_std']
    xgb_model = line_analyzer.fit_xgb(columns_to_ignore=columns_to_ignore, scale=False, test_size=0.5)
    #%%
    line_analyzer.plot_correlations()
    #%%
    line_analyzer.reduce_dimensions()
