def add_composition(df, formula_col_name ="formula"):
    from matminer.featurizers.conversions import StrToComposition
    df = StrToComposition().featurize_dataframe(df, formula_col_name)
    return df

def add_element_fraction(df, composition_col_name="composition"):
    from matminer.featurizers.composition.element import ElementFraction
    ef = ElementFraction()
    df = ef.featurize_dataframe(df, composition_col_name)
    return df

def add_magpie(df, composition_col_name="composition"):
    from matminer.featurizers.composition import ElementProperty
    magpie_properties = ElementProperty.from_preset(preset_name='magpie')
    df = magpie_properties.featurize_dataframe(df, "composition")
    return df