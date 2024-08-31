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

def add_valence_orbitals(df, composition_col_name="composition"):
    from matminer.featurizers.composition.orbital import ValenceOrbital
    val_orb_properties = ValenceOrbital()
    df = val_orb_properties.featurize_dataframe(df, "composition")
    return df

def add_Wen_alloys(df, composition_col_name="composition"):
    from matminer.featurizers.composition.alloy import WenAlloys
    wen_alloy_properties = WenAlloys()
    df = wen_alloy_properties.featurize_dataframe(df, "composition")
    return df

def add_Yang_alloys(df, composition_col_name="composition"):
    from matminer.featurizers.composition.alloy import YangSolidSolution
    yang_alloy_properties = YangSolidSolution()
    df = yang_alloy_properties.featurize_dataframe(df, "composition")
    return df

def add_Meredig(df, composition_col_name="composition"):
    from matminer.featurizers.composition import Meredig
    meredig_properties = Meredig()
    df = meredig_properties.featurize_dataframe(df, "composition")
    return df

