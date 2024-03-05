
import jiwer
import pywer

def calculate_wer(reference, hypothesis):
	ref_words = reference.split()
	hyp_words = hypothesis.split()
	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)

	# Total number of words in the reference text
	total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)


	wer = (substitutions + deletions + insertions) / total_words
	return wer



reference = "this is my first"
hypothesis = "this is my ss second"
ref_words=reference.split()
 
hyp_words = hypothesis.split()

print(sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp))
jiwer_error = jiwer.wer(reference, hypothesis)
pywer_error=pywer.wer(reference, hypothesis)
print("JIWER : {}".format(jiwer_error) )
print("PYWER : {}".format(pywer_error))
print("calculate_wer :{}".format(calculate_wer(reference, hypothesis)))
