from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
from queue import Queue
import unicodedata
import re
import logging
import streamlit as st
import io
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to initialize a WebDriver instance
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920x1080')
    
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# Function to normalize strings
def normalize_string(s):
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = s.lower()
    s = s.rstrip('.')
    s = re.sub(r'[-\s]+', ' ', s)
    return s.strip()

# Helper function to extract the number of resources from the detailed page
def extract_subject_of_works_count(driver):
    try:
        # Extract content from the detailed page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        subject_of_works_element = soup.find('div', id='related-resources-box-subjectof-count')
        if subject_of_works_element:
            text = subject_of_works_element.get_text(strip=True)
            logging.info(f"Extracted text: {text}")  # Debugging line to log the extracted text
            match = re.search(r'(\d+)\s+resources', text)
            if match:
                return int(match.group(1))
        else:
            logging.info("subject_of_works_element not found")  # Log if the element is not found
        return 0
    except Exception as e:
        logging.error(f"An error occurred while extracting subject of works count: {e}")
        return 0

# Update the scraping functions
def scrape_term(term, driver_queue):
    driver = driver_queue.get()
    try:
        subjects_url = "https://id.loc.gov/authorities/subjects.html"
        driver.get(subjects_url)
        wait = WebDriverWait(driver, 10)
        search_input = wait.until(EC.visibility_of_element_located(
            (By.XPATH, '//input[@type="search" and @placeholder="Search Library of Congress Subject Headings"]')
        ))
        search_input.clear()
        search_input.send_keys(f'"{term}"')
        search_input.send_keys(Keys.RETURN)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        matching_links = soup.find_all('a', {'title': 'Click to view record'})
        normalized_term = normalize_string(term)
        for link in matching_links:
            link_text = link.text.strip()
            normalized_link_text = normalize_string(link_text)
            if normalized_link_text == normalized_term:
                href = link.get('href')
                full_url = urljoin(subjects_url, href)
                
                # Navigate to the detailed page
                driver.get(full_url)
                time.sleep(3)  # Wait for the detailed page to load
                subject_of_works_count = extract_subject_of_works_count(driver)
                
                return {'search_terms': term, 'uri': full_url, '# resources of work': subject_of_works_count}
        return scrape_term_in_names(term, driver)
    except Exception as e:
        logging.error(f"An error occurred for term '{term}': {e}")
        return {'search_terms': term, 'uri': None, '# resources of work': 0}
    finally:
        driver_queue.put(driver)

# Update the function to search the term in names.html
def scrape_term_in_names(term, driver):
    try:
        names_url = "https://id.loc.gov/authorities/names.html"
        driver.get(names_url)
        wait = WebDriverWait(driver, 10)
        search_input = wait.until(EC.visibility_of_element_located(
            (By.XPATH, '//input[@type="search" and @placeholder="Search Library of Congress Name Authority File"]')
        ))
        search_input.clear()
        search_input.send_keys(f'"{term}"')
        search_input.send_keys(Keys.RETURN)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        matching_links = soup.find_all('a', {'title': 'Click to view record'})
        normalized_term = normalize_string(term)
        for link in matching_links:
            link_text = link.text.strip()
            normalized_link_text = normalize_string(link_text)
            if (normalized_link_text == normalized_term) and link.get('href'):
                href = link.get('href')
                full_url = urljoin(names_url, href)
                
                # Navigate to the detailed page
                driver.get(full_url)
                time.sleep(3)  # Wait for the detailed page to load
                subject_of_works_count = extract_subject_of_works_count(driver)
                
                return {'search_terms': term, 'uri': full_url, '# resources of work': subject_of_works_count}
        logging.info(f"No exact match found for term '{term}' in names.")
        return {'search_terms': term, 'uri': None, '# resources of work': 0}
    except Exception as e:
        logging.error(f"An error occurred for term '{term}' in names: {e}")
        return {'search_terms': term, 'uri': None, '# resources of work': 0}

# Function to process a chunk of data
def process_chunk(df_chunk, processed_count, progress_bar, status_placeholder):
    search_terms = df_chunk['search_terms'].tolist()[0:800]
    driver_queue = Queue()
    num_drivers = 5
    for _ in range(num_drivers):
        driver_queue.put(init_driver())

    results = []
    term_to_uri = {}
    term_to_resources_count = {}
    try:
        with ThreadPoolExecutor(max_workers=num_drivers) as executor:
            unique_terms = list(set(search_terms))
            futures = {executor.submit(scrape_term, term, driver_queue): term for term in unique_terms}
            for future in as_completed(futures):
                result = future.result()
                if result['uri']:
                    term_to_uri[result['search_terms']] = result['uri']
                    term_to_resources_count[result['search_terms']] = result['# resources of work']
                processed_count += 1
                if processed_count % 200 == 0:
                    logging.info(f"Completed processing {processed_count} search terms")
                progress_bar.progress(processed_count / len(search_terms))
                status_placeholder.text(f"Processed {processed_count} / {len(search_terms)} terms")
    finally:
        while not driver_queue.empty():
            driver = driver_queue.get()
            driver.quit()

    results_df = pd.DataFrame(list(term_to_uri.items()), columns=['search_terms', 'uri'])
    results_df['# resources of work'] = results_df['search_terms'].map(term_to_resources_count)
    df_chunk = df_chunk.merge(results_df, on='search_terms', how='left')
    return df_chunk, processed_count

# Function to save results to Excel
def save_results(all_results, save_path):
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_excel(save_path, index=False)
    logging.info(f"Saved results to {save_path}")

# Streamlit code for creating the interface
@st.cache_data
def cached_process(df):
    chunk_size = 800
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    all_results = []
    processed_count = 0

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    for chunk in chunks:
        processed_chunk, processed_count = process_chunk(chunk, processed_count, progress_bar, status_placeholder)
        all_results.append(processed_chunk)
        save_results(all_results, "processed_results.xlsx")

    final_df = pd.concat(all_results, ignore_index=True)
    return final_df

# Function to count URIs that match specific patterns
def count_matching_uris(df):
    name_pattern = r"https://id.loc.gov/authorities/names/n"
    subject_pattern = r"https://id.loc.gov/authorities/subjects/sh"

    # Count occurrences of each pattern
    name_count = df['uri'].apply(lambda x: bool(re.match(name_pattern, str(x)))).sum()
    subject_count = df['uri'].apply(lambda x: bool(re.match(subject_pattern, str(x)))).sum()

    return name_count, subject_count

def main():
    st.title("Web Scraper")
    
    # File uploader for the Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is None:
        st.markdown('<p style="color:red;">* Please upload an XLSX file with the only required column: search_terms</p>', unsafe_allow_html=True)

    if uploaded_file is not None:
        
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        
        # Process the Excel file using cached process function
        final_df = cached_process(df)

        # Selectable columns
        columns_to_display = st.multiselect("Select columns to display", final_df.columns.tolist(), default=final_df.columns.tolist())
        final_df = final_df[columns_to_display]
        
        # Single filter box for all columns - contains - not exact match. 
        filter_value = st.sidebar.text_input("Filter Results")
        
        if filter_value:
            mask = final_df.apply(lambda row: row.astype(str).str.contains(filter_value, case=False).any(), axis=1)
            final_df = final_df[mask]

        # Display the results
        st.write("Processing complete. Here are the results:")
        st.dataframe(final_df)

        # Count the URIs matching specific patterns
        name_count, subject_count = count_matching_uris(final_df)

        # Display the counts
        st.write(f"Number of Name Authority URIs: {name_count}")
        st.write(f"Number of Subject Heading URIs: {subject_count}")

        # Create a downloadable file
        buffer = io.BytesIO()
        final_df.to_excel(buffer, index=False)
        buffer.seek(0)

        # Download link for the processed results
        st.download_button(
            label="Download Filtered Results",
            data=buffer,
            file_name='processed_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        # Visualizations section
        st.title("Visualizations")
        st.subheader("Count of URI")  
        
        # Count the number of URIs found and the number of No URIs found
        uri_count = final_df['uri'].notna().sum()  # Count of rows where 'uri' is not null
        no_uri_count = final_df['uri'].isna().sum()  # Count of rows where 'uri' is null

        


        bar_chart_data = pd.DataFrame({
            'Category': ['URIs Found', 'No URIs Found'],
            'Count': [uri_count, no_uri_count]
        })

        # Display the bar chart
        st.bar_chart(bar_chart_data.set_index('Category'))

        # Expander for URIs found
        with st.expander("Download Rows with URIs"):
            uri_found_df = final_df[final_df['uri'].notna()]
            buffer_uri = io.BytesIO()
            uri_found_df.to_excel(buffer_uri, index=False)
            buffer_uri.seek(0)
            st.download_button(
                label="Download Rows with URIs",
                data=buffer_uri,
                file_name='rows_with_uris.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            image = cv2.imread('images/1.JPG')
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to grayscale (if it's not already a binary mask)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Apply a threshold to make sure the mask is binary (black and white)
            _, mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)        
                

            # Generate and display the word cloud for search terms with URIs
            st.subheader("Word Cloud for Search Terms with URIs")
            uri_terms = " ".join(uri_found_df['search_terms'].dropna())
            uri_wordcloud = WordCloud(width=500, height=500, background_color='white', contour_color="black", contour_width=3,min_font_size=3, mask=mask).generate(uri_terms)

            # Plot the word cloud
            plt.figure(figsize=(6, 3))
            plt.imshow(uri_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        # Expander for No URIs found
        with st.expander("Download Rows without URIs"):
            no_uri_df = final_df[final_df['uri'].isna()]
            buffer_no_uri = io.BytesIO()
            no_uri_df.to_excel(buffer_no_uri, index=False)
            buffer_no_uri.seek(0)
            st.download_button(
                label="Download Rows without URIs",
                data=buffer_no_uri,
                file_name='rows_without_uris.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            # Generate and display the word cloud for search terms without URIs
            st.subheader("Word Cloud for Search Terms without URIs")
            no_uri_terms = " ".join(no_uri_df['search_terms'].dropna())
            no_uri_wordcloud = WordCloud(width=500, height=500, background_color='white', contour_color="black",contour_width=3,min_font_size=3, mask = mask).generate(no_uri_terms)
            plt.figure(figsize=(6, 3))
            plt.imshow(no_uri_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)


        

        # **Interactive Horizontal Bar Chart for Search Terms vs # Resources of Work**
        st.subheader("Search Terms vs. Number of Resources of Work")
        terms_resources_df = final_df[['search_terms', '# resources of work']].dropna().sort_values(by='# resources of work', ascending=False)

        fig = px.bar(
            terms_resources_df,
            x='# resources of work',
            y='search_terms',
            orientation='h',
            labels={'# resources of work': 'Number of Resources', 'search_terms': 'Search Terms'},
            text='# resources of work',
            height=600
        )

        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig)
        st.write("Less Resources of work indicate opportunity for the Cataloguing and MetaData Department to add more resources relating to that particular subject heading or name authority")


if __name__ == "__main__":
    main()

