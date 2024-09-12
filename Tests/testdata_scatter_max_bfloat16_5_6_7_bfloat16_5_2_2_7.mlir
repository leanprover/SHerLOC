"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<5x2x2x7xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2x1xi64>, tensor<5x2x2x7xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<5x2x2x7xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xEF4075BF57BE8D3F16C05840814060BE3E3F98BF89C024C0A73FECBFA03F9AC030BFC2BEE93FFF3F97C01EBE683F1FC0BDBF35C071C082C08D4009C0184010BE90C0DFBF8A409740DFBF06C034C0AE3FBDBFBB4083C03440413F3840074013C086BD88BF47BF29C0A8BF5E3FAE3D1140A940C7BF9BC083C08FC0373F39404FC026C0BEBE913F8340AABFDC40913EAA40DABF08404140E5C093BFAFBF32C0FE3FA240DD3F94BFA73F3CC097BFBE3FEEC09FC0A540583EBBBF9A4025C06D3ECABE39C0C6BF75C00FC0203FAF4014C1CA4082BF9C3F084050407A40B4BEB03F843F2D4082C090C08340A3BF45C0993F1E3FA140D0BF073F7D404A3FC9BE57BE0CBF4940EF3E4FBFCCBE2DC088BEDABF494086405FC08CC0BA3F98C079C0564052C0F9BF18C08A3FF63F49BEDA3F554049403F4076C006C1F7BD173E36402DC06BC005C0B23F5EBFCBC0803F62C09D401CC024C0B73FC93F14C0943F264030C01BBEC63F6DBEF43EA4409DBF40C082BE9ABF13407CC0114075C0B2BF0A40CCBFDEBE0A40FC3FC8BF584049BF5BBE7D40B0BF9CBE2C4012C0EABF4440ACC0823C6F405340533F"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<"0x5840DFBCC4C0FBBDC1BF01BF46C0823EAFBF2040183FB3C061C0984066C0AEBF0A4004C072400EBF0040703E29409DC02CC01CC066C0BDBEFF3E313F52BF9B402DBE72C007C05940C13D354022407F3DD5C0E7BFAE3FD5BF403F604076BF583FFABF5B3E6C3F89406F3FA3C039C0423FE4408340164025C054BF2D3F9E3E6AC05A3F24BF264014BF41407FC085BF83BFD4BE2D40A4BF7B4022C0D0BF41C0AD40B640D9BFC5C0E23F31C06340E3BF31400340A7C086C01BC046C0193EF2C0EE3EA33F8B3F0CC0CFBED2BFB3C0B4BF50BF1B406AC07E3EA2C0E7BEFA3F74C01BC040C0D5BE1EC0C1C0324080C0AF400B401DBFF7C08DC057BFAD40503E1EC006408CC0AC3F3B3EA93F144087BE6C40DEBF12C1923F8CC04F40"> : tensor<5x2x2x7xbf16>}> : () -> tensor<5x2x2x7xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<5x2x2x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xEF40DFBC57BE8D3FC1BF58408140823E3E3F2040183F24C0A73F9840A03FAEBF0A40C2BE7240FF3F0040703E29401FC0BDBF1CC066C0BDBE8D4009C0184010BE90C0DFBF8A409740DFBF06C034C0AE3FBDBFBB40FF3E3440413F9B40074013C086BD5940C13D354022405E3FAE3D1140A940C7BF403F604076BF583F39405B3E6C3F8940913F8340AABFDC40913EAA40DABF08404140E5C093BFAFBF32C0FE3FA240DD3F94BFA73FE4408340164025C054BFA5409E3EBBBF9A4024BF2640CABE4140C6BF85BF83BF203FAF40A4BFCA4082BF9C3F0840AD40B640B4BEB03FE23F2D4082C090C08340A3BF45C0993F1E3FA140D0BF073F7D404A3FC9BE57BE6340494031400340CCBE2DC088BEDABF49408640EE3EA33FBA3F0CC0CFBE564052C0B4BF50BF1B40F63F7E3EDA3F554049403F401BC006C1F7BD173E36402DC06BC005C0B23F5EBFCBC0803F62C09D401CC024C0B73FC93F14C032402640AF400B40C63F6DBEF43EA440AD40503E82BE06401340AC3F1140A93F14400A406C40DEBE0A40FC3FC8BF584049BF5BBE7D40B0BF9CBE2C4012C0EABF4440ACC0823C6F405340533F"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

