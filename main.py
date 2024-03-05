import speech_recognition as sr
from pydub import AudioSegment
import pyttsx3
from vosk import Model, KaldiRecognizer
import wave
import json
import deepspeech
import numpy as np
from jiwer import wer
import os
# from pocketsphinx import AudioFile, get_model_path
from google.cloud import speech_v1p1beta1 as speech
import assemblyai as aai
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings


english_reference_text="study says fructose is a major cause of obesity this is breaking news english dot com Scientists have discovered that fructose, a naturally occurring sugar, is a major driver of obesity. Fructose is also known as fruit sugar. It occurs to varying degrees in fruit and vegetables. It is also used in processed form in high fructose corn syrup, which is in a lot of the food we eat, and promotes obesity. A study led by Dr Richard Johnson at the University of Colorado found that although fructose isn't the biggest source of calorific intake, it stimulates an urge to eat fattier food. Researchers posited a shift of focus on what we eat. They wrote: All hypotheses recognize the importance of reducing 'junk' foods, [however] it remains unclear whether the focus should be on reducing [fructose] intake. Dr Johnson and his colleagues conducted an exhaustive study of all known contributors to obesity. They found that the process of our body converting fructose into energy causes a drop in the levels of a compound called ATP. When ATP falls, our body tells us to eat more. Researchers call this process the fructose survival hypothesis. Johnson said: Fructose is what triggers our metabolism to go into low power mode and lose our control of appetite, but fatty foods become the major source of calories that drive weight gain. Scientists have attributed the consumption of high amounts of fructose to health issues. The most common of these is non-alcoholic fatty liver disease."
german_reference_text= "spracherkennung ist der Prozess bei dem ein audiosignal mithilfe eines computerprogramms in text übersetzt wirdautomatische Spracherkennung ist die kunst gesprochene Sprache oder ein audiosignal in geschriebenen Text mithilfe eines programms umzuwandeln hat in den letzten jahrzehnten sehr viele fortschritte geschafft"
class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_audio(self, file_paths, library):
        if library == 'vosk':
           return self.recognize_audio_vosk(file_paths)
        elif library == 'speech_recognition':
           return self.recognize_audio_speech_recognition(file_paths)
        elif library == 'deepspeech_mozilla':
           return self.recognize_audio_deepspeech_mozilla(file_paths)
        # elif library == 'pocketsphinx':
        #    return self.recognize_audio_pocketsphinx(file_paths)assemblyai
        elif library == 'assemblyai':
           return self.recognize_audio_assemblyai(file_paths)
        elif library == 'google_cloud':
           return self.recognize_audio_google_cloud(file_paths)
        else:
           raise ValueError("Invalid library name")
    #speech_recognition
    def recognize_audio_speech_recognition(self,files):
        library_name = 'speech_recognition'
        output = {library_name: {}}
        results = []
        for i, file_path in enumerate(files):

        # Check the file format and convert to WAV if needed
            if not file_path.endswith(".wav"):
                audio = AudioSegment.from_file(file_path, format="mp3")
                file_path = "temp.wav"
                audio.export(file_path, format="wav")

            # Load the audio file
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)

            if i == 0 :
                language='German'
                code_language = "nl-NL"
                reference_text=german_reference_text
            else :
                language='Engish'
                code_language= "en-US"
                reference_text=english_reference_text

            try:
                # Use the recognize_google function to convert the audio to text
                text = self.recognizer.recognize_google(audio_data, language=code_language)

                if language not in output[library_name]:
                    output[library_name][language] = []

                output[library_name][language].append(str(wer(reference_text, text)))

            except sr.UnknownValueError:
                results.append("Speech recognition could not understand the audio content.")
            except sr.RequestError:
                results.append("There was an issue with the API request.")


        return output

    def recognize_audio_vosk(self, files):
        library_name = 'vosk'
        output = {library_name: {}}

        for i, file_path in enumerate(files):
            if i == 0:
                model = Model("models_vosk/vosk-model-small-de-0.15")
                language = 'German'
                reference_text = german_reference_text
            else:
                model = Model("models_vosk/vosk-model-small-en-us-0.15")
                language = 'English'
                reference_text = english_reference_text

            kaldi_recognizer = KaldiRecognizer(model, 16000)

            with wave.open(file_path, 'rb') as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                    return "Audio file must be mono PCM."

                recognized_text = ""
                while True:
                    data = wf.readframes(8192)
                    if len(data) == 0:
                        break

                    if kaldi_recognizer.AcceptWaveform(data):
                        result_text = kaldi_recognizer.FinalResult()
                        data_dict = json.loads(result_text)
                        text = data_dict["text"]
                        recognized_text += text + " "

                if language not in output[library_name]:
                    output[library_name][language] = []

                wer_value = str(wer(reference_text, recognized_text))
                output[library_name][language].append({"WER": wer_value, "Text": recognized_text})

        return output

    def recognize_audio_deepspeech_mozilla(self, files):
        library_name = 'deepspeech_mozilla'
        output = {library_name: {}}

        for i, file_path in enumerate(files):

            if i == 0:
                model_file_path = "models_deep_speech_mozilla/output_graph_de.pbmm"
                language = 'German'
                reference_text = german_reference_text
            else:
                model_file_path = "models_deep_speech_mozilla/deepspeech-0.9.3-models.pbmm"
                language = 'English'
                reference_text = english_reference_text

            model = deepspeech.Model(model_file_path)

            stream = model.createStream()

            buf = bytearray(1024)
            with open(file_path, 'rb') as audio:
                while audio.readinto(buf):
                    data16 = np.frombuffer(buf, dtype=np.int16)
                    stream.feedAudioContent(data16)

            text = stream.finishStream()
            if language not in output[library_name]:
                output[library_name][language] = []

            wer_value = str(wer(reference_text, text))
            output[library_name][language].append({"WER": wer_value, "Text": text})

        return output

    # def recognize_audio_pocketsphinx(self, files):
    #     library_name = 'pocketsphinx'
    #     output = {library_name: {}}
    #     for i, file_path in enumerate(files):
    #         model_path = get_model_path()
    #         config = {
    #             'verbose': False,
    #             'audio_file': file_path,
    #             'hmm': os.path.join(model_path, 'en-us'),
    #             'lm': os.path.join(model_path, 'en-us.lm.bin'),
    #             'dict': os.path.join(model_path, 'cmudict-en-us.dict')
    #         }
    #         audio = AudioFile(**config)
    #         phrases = [str(phrase) for phrase in audio]
    #         text = ' '.join(phrases)
    #         if i == 0 :
    #             language='German'
    #             reference_text=german_reference_text
    #         else :
    #             language='Engish'
    #             reference_text=english_reference_text
    #         if language not in output[library_name]:
    #             output[library_name][language] = []
    #         output[library_name][language].append(str(wer(reference_text, text)))
    #     return output

    def recognize_audio_google_cloud(self, files):

        library_name = 'google_cloud'
        output = {library_name: {}}

      
        key_path = "key.json"


        client = speech.SpeechClient.from_service_account_file(key_path)

        for i, file_path in enumerate(files):

            if i == 0:
                language = 'German'
                code_language = "de-DE"
                reference_text = german_reference_text
            else:
                language = 'English'
                code_language = "en-US"
                reference_text = english_reference_text

     
            gcs_uri = f"gs://user_bucket/{file_path.split('/', 1)[1]}"
            audio_file = speech.RecognitionAudio(uri=gcs_uri)

       
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=code_language,
                enable_automatic_punctuation=True,
            )

            response = client.long_running_recognize(config=config, audio=audio_file)

        
            response = response.result()

          
            text = ""
            for result in response.results:
                text += result.alternatives[0].transcript

            if language not in output[library_name]:
                output[library_name][language] = []

            wer_value = str(wer(reference_text, text))
            output[library_name][language].append({"WER": wer_value, "Text": text})

        return output

    def recognize_audio_assemblyai(self, files):
        library_name = 'assemblyai'
        output = {library_name: {}}

        for i, file_path in enumerate(files):
            if i == 0:
                language = 'German'
                language_code = 'de'
                reference_text = german_reference_text
            else:
                language = 'English'
                language_code = 'en_us'
                reference_text = english_reference_text

            aai.settings.api_key = "API_KEY "
            transcriber = aai.Transcriber()
            config = aai.TranscriptionConfig(language_code=language_code)
            transcript = transcriber.transcribe(file_path, config=config)

            if language not in output[library_name]:
                output[library_name][language] = []

            wer_value = str(wer(reference_text, transcript.text))
            output[library_name][language].append({"WER": wer_value, "Text": transcript.text})

        return output
    def write_wers_to_file(self, wers, output_file):
        with open(output_file, "w") as file:
            for library_name, languages in wers.items():
                file.write(f"Library: {library_name}\n")
                for language, results in languages.items():
                    file.write(f"Language: {language}\n")
                    for i, result in enumerate(results):
                        wer_value = result["WER"]
                        recognized_text = result["Text"]
                        file.write(f"Result {i + 1}: WER: {wer_value}, Recognized Text: {recognized_text}\n")
                    file.write("\n")
                file.write("\n")

if __name__ == "__main__":
    processor = AudioProcessor()
    libraries = ['vosk', 'deepspeech_mozilla', 'assemblyai', 'google_cloud',]

    combined_results = {}  # Dictionary to store combined results

    while libraries:
        # Display library options
        print("Available Libraries:")
        for i, library in enumerate(libraries, start=1):
            print(f"{i}. {library}")

        # Prompt the user to choose a library
        selected_library_index = int(input("Choose a library (enter the corresponding number): ")) - 1

        try:
            selected_library = libraries[selected_library_index]

            # Test speech recognition with the selected library
            file_paths = ["audio/spracherkennung.wav", "audio/231023-fructose-and-obesity wav.wav"]
            wers = processor.recognize_audio(file_paths, selected_library)

            output_file = f"output/{selected_library}_recognized_results.txt"
            processor.write_wers_to_file(wers, output_file)

            print(f"Results saved to {output_file}")

         
            # Add the results to the combined_results dictionary
            combined_results[selected_library] = wers[selected_library]

            # Remove the selected library from the list
            libraries.remove(selected_library)

            # Ask the user if they want to continue with the next library
            continue_option = input("Do you want to continue with the next library? (yes/no): ").lower()
            if continue_option != 'yes':
                break  # Exit the loop if the user enters anything other than 'yes'

        except IndexError:
            print("Invalid library choice. Please enter a valid library number.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Save combined results to a file
    combined_output_file = "output/combined_recognized_results.txt"
    processor.write_wers_to_file(combined_results, combined_output_file)
    print(f"Combined results saved to {combined_output_file}")
 
