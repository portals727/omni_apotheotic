import tkinter as tk
from tkinter import filedialog, messagebox

def select_input_file():
    input_path = filedialog.askopenfilename(
        title="Select TXT Pulse Log File",
        filetypes=[("Text files", "*.txt")]
    )
    if input_path:
        input_label.config(text=f"Input: {input_path}")
        global selected_input
        selected_input = input_path

def select_output_file():
    output_path = filedialog.asksaveasfilename(
        title="Save BIN File As",
        defaultextension=".bin",
        filetypes=[("Binary files", "*.bin")]
    )
    if output_path:
        output_label.config(text=f"Output: {output_path}")
        global selected_output
        selected_output = output_path

def convert_file():
    try:
        if not selected_input or not selected_output:
            messagebox.showerror("Error", "Please select both input and output paths.")
            return

        with open(selected_input, 'r') as f:
            lines = f.readlines()

        with open(selected_output, 'wb') as out:
            for line in lines:
                if ':' in line and line.strip()[0].isdigit():
                    parts = line.split(':')[1].strip().split()
                    if len(parts) == 4:
                        for bin_str in parts:
                            val = int(bin_str, 2)
                            out.write(bytes([val]))

        messagebox.showinfo("Success", "Conversion completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Conversion failed: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("Pulse Log TXT to BIN Converter")
root.geometry("500x200")

selected_input = None
selected_output = None

input_button = tk.Button(root, text="Select Input TXT File", command=select_input_file)
input_button.pack(pady=10)

input_label = tk.Label(root, text="Input: None selected")
input_label.pack()

output_button = tk.Button(root, text="Select Output BIN Location", command=select_output_file)
output_button.pack(pady=10)

output_label = tk.Label(root, text="Output: None selected")
output_label.pack()

convert_button = tk.Button(root, text="Convert", command=convert_file)
convert_button.pack(pady=20)

root.mainloop()