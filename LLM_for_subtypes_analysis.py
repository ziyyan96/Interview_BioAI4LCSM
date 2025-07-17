#%%


# 每个 PAM50 亚型对应一个上调和下调的基因列表

prompts = {
    "LumA_up": """In breast cancer, the LumA subtype significantly upregulates the following genes compared to other subtypes: MAPT, UBXN10, WBP1L, NEK10, LRRC48.
What biological features or pathways might these genes represent?""",

    "LumA_down": """In breast cancer, the LumA subtype significantly downregulates the following genes compared to other subtypes: MYBL2, CDC45, CDCA5, PTTG1, AURKB, CENPA, ORC6, AURKA, PRR11, MCM10.
What suppressed biological activities or pathways might this imply for the LumA subtype?""",

    "LumB_up": """In breast cancer, the LumB subtype significantly upregulates the following genes compared to other subtypes: COX17, TIMELESS, HAGH, RACGAP1, CDC25C, GINS2, ESPL1, CCNB1, E2F1, BRCA1.
What biological features or pathways might these genes represent?""",

    "LumB_down": """In breast cancer, the LumB subtype significantly downregulates the following genes compared to other subtypes: TRIM29, SFRP1, GU228584, KRT5, KRT17, WWTR1, EGFR, BOC, RGMA, PTPN14.
What suppressed biological activities or pathways might this imply for the LumB subtype?""",

    "Normal_up": """In breast cancer, the Normal-like subtype significantly upregulates the following genes compared to other subtypes: HPSE2, DLK2, SPRY2, ADCYAP1R1, SAMD5, PPAP2B, FIGF, TSHZ2, PAMR1, SYNM.
What biological features or pathways might these genes represent?""",

    "Normal_down": """In breast cancer, the Normal-like subtype significantly downregulates the following genes compared to other subtypes: PAFAH1B3, COL11A1, APOA1BP, STXBP2, ESM1, MMP11.
What suppressed biological activities or pathways might this imply for the Normal-like subtype?""",

    "Basal_up": """In breast cancer, the Basal subtype significantly upregulates the following genes compared to other subtypes: FOXC1, LEMD1, DQ576756, A2ML1, UGT8, ROPN1, VGLL1, SFT2D2, PSAT1, TTLL4.
What biological features or pathways might these genes represent?""",

    "Basal_down": """In breast cancer, the Basal subtype significantly downregulates the following genes compared to other subtypes: FOXA1, MLPH, SPDEF, AR, SLC44A4, PRR15, GATA3, DEGS2, CMBL, C14orf25.
What suppressed biological activities or pathways might this imply for the Basal subtype?""",

    "Her2_up": """In breast cancer, the Her2 subtype significantly upregulates the following genes compared to other subtypes: STARD3, ERBB2, GRB7, MIEN1, PGAP3, TCAP, G6PD, PSMD3, ORMDL3, SOX11.
What biological features or pathways might these genes represent?""",

    "Her2_down": """In breast cancer, the Her2 subtype significantly downregulates the following genes compared to other subtypes: BCL2, IGF1R, DBNDD2, RERG, NUDT6, ESR1, ASB13, SUSD3, MED13L, PCSK6.
What suppressed biological activities or pathways might this imply for the Her2 subtype?""",
}
#%%
import openai

openai.api_key = "your-api-key"

for name, prompt in prompts.items():
    print(f"\n=== {name} ===")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    print(response['choices'][0]['message']['content'])
