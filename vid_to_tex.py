from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

OPENAI_API_KEY = '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'


loader = YoutubeLoader.from_youtube_url("https://youtu.be/WQdqgrWvy6g?si=S_LGjgatxexcRgX9", add_video_info=True)
result = loader.load()

print (f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")
print ("")
# print (result)

def create_pdf_from_data(result, output_pdf_filename):
    # Check if the PDF file already exists, and remove it if it does
    if os.path.exists(output_pdf_filename):
        os.remove(output_pdf_filename)

    # Create a PDF canvas
    c = canvas.Canvas(output_pdf_filename, pagesize=letter)

    # Define the initial Y-coordinate for content
    y_position = 700

    # Iterate through each 'item' in the 'result'
    for item in result:
        # Add content to the PDF
        c.drawString(100, y_position, f"Author: {item.metadata['author']}")
        y_position -= 20  # Adjust the Y-coordinate for the next line
        c.drawString(100, y_position, f"Length: {item.metadata['length']} seconds")
        y_position -= 20  # Adjust the Y-coordinate for the next line
        c.drawString(100, y_position, f"Title: {item.metadata['title']}")
        y_position -= 20
        # Add more content as needed
        # Check if 'item.page_content' is not None and add it to the PDF
        if item.page_content:
            y_position -= 20  # Adjust the Y-coordinate for the next line
            content_lines = item.page_content.split('\n')
            for line in content_lines:
                # Ensure that the text wraps within the page width
                if y_position < 50:  # Start a new page if close to the bottom
                    c.showPage()
                    y_position = 700  # Reset Y-coordinate for new page
                c.drawString(100, y_position, line)
                y_position -= 20  # Adjust the Y-coordinate for the next line

        # Move to the next page if there is more content
        c.showPage()

    # Save the PDF file
    c.save()

# Specify the output PDF filename
output_pdf_filename = 'Video_out.pdf'

# Call the function to create the PDF file
create_pdf_from_data(result, output_pdf_filename)

print(f"PDF file '{output_pdf_filename}' has been created.")