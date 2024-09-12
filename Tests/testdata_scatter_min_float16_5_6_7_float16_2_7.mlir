"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<2x7xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2xi64>, tensor<2x7xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<2x7xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x72C4EDA6F6BCFE34BBB858C490C448C423B6A33F90B9E53DA0ACDBC478C42EC144C48141ADC3F3357E41FA4669BAC9B43ABAF0C03F416E3DC7323BC123C2AEBDFF4336B6FB466543B0389B2C85B45CB174B033C2C94053BF14B69F3EB33C7040AC3FB5459D467543763B59425B3FC7B78140EBBE333C8EBB3846924470AF97BF71BAA942E4C0CC3C24C0863634C193C439422B432E3432442944B4C4C7C361C386BE89451BB8F5B68DC255C14A31E23899C1313C27C3F638E33AD43FE642D9BD83B669C496C26D3B90422DC08A41D23C0DB474C40A40064240C4944327402539743DDF3D174323C287C58C3C6F3C24BE34B938438EB93A3EB63C22C30DC44448C8BEB643CDB6773E6FC264353B409B3B9ABF81430844CE4060C109C392337F46D1B1CEB8B6B3BA38C13BB1C45FB894C115B564C5C8442BC1753A57C5D4BC37434BC2F8C099B8E443EA44FBC33C3FF2BA39C057C88FC173C0193C4CC561C2C9381E3D2C355B3D2B3EE1B732C033BF78407741293FCCC31BB80ABD68BBB3BBEEB9B643D7B5C2C17B42E33F1E3C8E3FF2B963BB5644AD3E47C04E40F2B99EBE76C5DB402241"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[2.464840e+00, 1.959960e+00, -6.625000e+00, 5.644530e+00, 7.719720e-01, 2.089840e+00, 1.393550e+00], [1.035160e+00, 2.677730e+00, 3.195310e+00, 1.890630e+00, -2.314450e+00, 3.482420e+00, 1.361080e-01]]> : tensor<2x7xf16>}> : () -> tensor<2x7xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<2x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x72C4EDA6F6BCFE34BBB858C490C448C423B6A0C690B92D3AA0ACDBC478C42EC144C48141ADC3F3357E41FA4669BAC9B43ABAF0C03F416E3DC7323BC123C2AEBDFF4336B6FB466543B0389B2C85B45CB174B033C2C94053BF14B69F3EB33C7040AC3FB5459D467543763B59425B3FC7B78140EBBE333C8EBB3846924470AF97BF71BAA942E4C0CC3C24C0863634C193C439422B432E3432442944B4C4C7C361C386BE89451BB8F5B68DC255C14A31E23899C1313C27C3F638E33AD43FE642D9BD83B669C496C26D3B90422DC08A41D23C0DB474C40A40064240C4A1C027405B30743DDF3D174323C287C58C3C6F3C24BE34B938438EB93A3EB63C22C30DC44448C8BEB643CDB6773E6FC264353B409B3B9ABF81430844CE4060C109C392337F46D1B1CEB8B6B3BA38C13BB1C45FB894C115B564C5C8442BC1753A57C5D4BC37434BC2F8C099B8E443EA44FBC33C3FF2BA39C057C88FC173C0193C4CC561C2C9381E3D2C355B3D2B3EE1B732C033BF78407741293FCCC31BB80ABD68BBB3BBEEB9B643D7B5C2C17B42E33F1E3C8E3FF2B963BB5644AD3E47C04E40F2B99EBE76C5DB402241"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

