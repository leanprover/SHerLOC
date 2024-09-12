"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %141 = "func.call"() <{callee = @inputs}> : () -> tensor<5x4xbf16>
    %142 = "func.call"() <{callee = @expected}> : () -> tensor<5xi64>
    %143 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %144 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %145 = "stablehlo.shift_right_logical"(%143, %144) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %146 = "stablehlo.convert"(%145) : (tensor<i64>) -> tensor<ui32>
    %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %148 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %149 = "stablehlo.convert"(%148) : (tensor<ui32>) -> tensor<i64>
    %150 = "stablehlo.and"(%143, %149) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %151 = "stablehlo.convert"(%150) : (tensor<i64>) -> tensor<ui32>
    %152 = "stablehlo.broadcast_in_dim"(%151) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %153 = "stablehlo.concatenate"(%147, %152) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %154 = "func.call"(%153) <{callee = @_gumbel}> : (tensor<2xui32>) -> tensor<5x4xbf16>
    %155 = "stablehlo.add"(%154, %141) : (tensor<5x4xbf16>, tensor<5x4xbf16>) -> tensor<5x4xbf16>
    %156 = "func.call"(%155) <{callee = @argmax}> : (tensor<5x4xbf16>) -> tensor<5xi64>
    "stablehlo.custom_call"(%156, %142) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5xi64>, tensor<5xi64>) -> ()
    "func.return"(%156) : (tensor<5xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %140 = "stablehlo.constant"() <{value = dense<[[-1.257810e+00, -1.625000e+00, -2.921880e+00, -2.296880e+00], [-7.304680e-01, -4.968750e+00, 3.343750e+00, -5.976560e-01], [-4.863280e-01, 4.468750e+00, -3.218750e+00, 2.906250e+00], [-3.312500e+00, 2.562500e+00, -2.265630e+00, -1.804690e+00], [-1.406250e+00, 7.070310e-01, -1.367190e+00, -3.312500e+00]]> : tensor<5x4xbf16>}> : () -> tensor<5x4xbf16>
    "func.return"(%140) : (tensor<5x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %139 = "stablehlo.constant"() <{value = dense<[3, 2, 1, 1, 1]> : tensor<5xi64>}> : () -> tensor<5xi64>
    "func.return"(%139) : (tensor<5xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<5x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_gumbel", sym_visibility = "private"}> ({
  ^bb0(%arg34: tensor<2xui32>):
    %132 = "stablehlo.constant"() <{value = dense<1.175490e-38> : tensor<bf16>}> : () -> tensor<bf16>
    %133 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %134 = "func.call"(%arg34, %132, %133) <{callee = @_uniform}> : (tensor<2xui32>, tensor<bf16>, tensor<f64>) -> tensor<5x4xbf16>
    %135 = "stablehlo.log"(%134) : (tensor<5x4xbf16>) -> tensor<5x4xbf16>
    %136 = "stablehlo.negate"(%135) : (tensor<5x4xbf16>) -> tensor<5x4xbf16>
    %137 = "stablehlo.log"(%136) : (tensor<5x4xbf16>) -> tensor<5x4xbf16>
    %138 = "stablehlo.negate"(%137) : (tensor<5x4xbf16>) -> tensor<5x4xbf16>
    "func.return"(%138) : (tensor<5x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<bf16>, tensor<f64>) -> tensor<5x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg13: tensor<2xui32>, %arg14: tensor<bf16>, %arg15: tensor<f64>):
    %68 = "stablehlo.convert"(%arg15) : (tensor<f64>) -> tensor<bf16>
    %69 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<1x1xbf16>
    %70 = "stablehlo.broadcast_in_dim"(%68) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<1x1xbf16>
    %71 = "stablehlo.constant"() <{value = dense<0> : tensor<1xui32>}> : () -> tensor<1xui32>
    %72 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<5xui32>
    %73 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %74 = "stablehlo.reshape"(%73) : (tensor<1xui32>) -> tensor<ui32>
    %75 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %76 = "stablehlo.reshape"(%75) : (tensor<1xui32>) -> tensor<ui32>
    %77 = "stablehlo.concatenate"(%72, %71) <{dimension = 0 : i64}> : (tensor<5xui32>, tensor<1xui32>) -> tensor<6xui32>
    %78 = "stablehlo.slice"(%77) <{limit_indices = array<i64: 3>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<6xui32>) -> tensor<3xui32>
    %79 = "stablehlo.slice"(%77) <{limit_indices = array<i64: 6>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<6xui32>) -> tensor<3xui32>
    %80 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %81 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %82 = "stablehlo.xor"(%74, %76) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %83 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %84 = "stablehlo.xor"(%82, %83) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %85 = "stablehlo.broadcast_in_dim"(%74) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %86 = "stablehlo.add"(%78, %85) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %87 = "stablehlo.broadcast_in_dim"(%76) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %88 = "stablehlo.add"(%79, %87) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %89 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %90 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %91:9 = "stablehlo.while"(%90, %89, %86, %88, %76, %84, %74, %80, %81) ({
    ^bb0(%arg25: tensor<i64>, %arg26: tensor<i64>, %arg27: tensor<3xui32>, %arg28: tensor<3xui32>, %arg29: tensor<ui32>, %arg30: tensor<ui32>, %arg31: tensor<ui32>, %arg32: tensor<4xui32>, %arg33: tensor<4xui32>):
      %130 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %131 = "stablehlo.compare"(%arg25, %130) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%131) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg16: tensor<i64>, %arg17: tensor<i64>, %arg18: tensor<3xui32>, %arg19: tensor<3xui32>, %arg20: tensor<ui32>, %arg21: tensor<ui32>, %arg22: tensor<ui32>, %arg23: tensor<4xui32>, %arg24: tensor<4xui32>):
      %127:8 = "func.call"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24) <{callee = @None}> : (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %128 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %129 = "stablehlo.add"(%arg16, %128) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%129, %127#0, %127#1, %127#2, %127#3, %127#4, %127#5, %127#6, %127#7) : (tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %92 = "stablehlo.concatenate"(%91#2, %91#3) <{dimension = 0 : i64}> : (tensor<3xui32>, tensor<3xui32>) -> tensor<6xui32>
    %93 = "stablehlo.slice"(%92) <{limit_indices = array<i64: 5>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<6xui32>) -> tensor<5xui32>
    %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64: 1>}> : (tensor<5xui32>) -> tensor<1x5xui32>
    %95 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4x1xui32>
    %96 = "stablehlo.constant"() <{value = dense<8> : tensor<ui32>}> : () -> tensor<ui32>
    %97 = "stablehlo.broadcast_in_dim"(%96) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x1xui32>
    %98 = "stablehlo.multiply"(%97, %95) : (tensor<4x1xui32>, tensor<4x1xui32>) -> tensor<4x1xui32>
    %99 = "stablehlo.broadcast_in_dim"(%94) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x5xui32>) -> tensor<4x5xui32>
    %100 = "stablehlo.broadcast_in_dim"(%98) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<4x1xui32>) -> tensor<4x5xui32>
    %101 = "stablehlo.shift_right_logical"(%99, %100) : (tensor<4x5xui32>, tensor<4x5xui32>) -> tensor<4x5xui32>
    %102 = "stablehlo.constant"() <{value = dense<255> : tensor<ui32>}> : () -> tensor<ui32>
    %103 = "stablehlo.broadcast_in_dim"(%102) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x5xui32>
    %104 = "stablehlo.and"(%103, %101) : (tensor<4x5xui32>, tensor<4x5xui32>) -> tensor<4x5xui32>
    %105 = "stablehlo.transpose"(%104) <{permutation = array<i64: 1, 0>}> : (tensor<4x5xui32>) -> tensor<5x4xui32>
    %106 = "stablehlo.reshape"(%105) : (tensor<5x4xui32>) -> tensor<20xui32>
    %107 = "stablehlo.convert"(%106) : (tensor<20xui32>) -> tensor<20xui8>
    %108 = "stablehlo.reshape"(%107) : (tensor<20xui8>) -> tensor<5x4xui8>
    %109 = "stablehlo.convert"(%108) : (tensor<5x4xui8>) -> tensor<5x4xui16>
    %110 = "stablehlo.constant"() <{value = dense<1> : tensor<ui16>}> : () -> tensor<ui16>
    %111 = "stablehlo.broadcast_in_dim"(%110) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<5x4xui16>
    %112 = "stablehlo.shift_right_logical"(%109, %111) : (tensor<5x4xui16>, tensor<5x4xui16>) -> tensor<5x4xui16>
    %113 = "stablehlo.constant"() <{value = dense<16256> : tensor<ui16>}> : () -> tensor<ui16>
    %114 = "stablehlo.broadcast_in_dim"(%113) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<5x4xui16>
    %115 = "stablehlo.or"(%112, %114) : (tensor<5x4xui16>, tensor<5x4xui16>) -> tensor<5x4xui16>
    %116 = "stablehlo.bitcast_convert"(%115) : (tensor<5x4xui16>) -> tensor<5x4xbf16>
    %117 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %118 = "stablehlo.broadcast_in_dim"(%117) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<5x4xbf16>
    %119 = "stablehlo.subtract"(%116, %118) : (tensor<5x4xbf16>, tensor<5x4xbf16>) -> tensor<5x4xbf16>
    %120 = "stablehlo.subtract"(%70, %69) : (tensor<1x1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %121 = "stablehlo.broadcast_in_dim"(%120) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xbf16>) -> tensor<5x4xbf16>
    %122 = "stablehlo.multiply"(%119, %121) : (tensor<5x4xbf16>, tensor<5x4xbf16>) -> tensor<5x4xbf16>
    %123 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xbf16>) -> tensor<5x4xbf16>
    %124 = "stablehlo.add"(%122, %123) : (tensor<5x4xbf16>, tensor<5x4xbf16>) -> tensor<5x4xbf16>
    %125 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xbf16>) -> tensor<5x4xbf16>
    %126 = "stablehlo.maximum"(%125, %124) : (tensor<5x4xbf16>, tensor<5x4xbf16>) -> tensor<5x4xbf16>
    "func.return"(%126) : (tensor<5x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg5: tensor<i64>, %arg6: tensor<3xui32>, %arg7: tensor<3xui32>, %arg8: tensor<ui32>, %arg9: tensor<ui32>, %arg10: tensor<ui32>, %arg11: tensor<4xui32>, %arg12: tensor<4xui32>):
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.add"(%arg5, %13) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %16 = "stablehlo.reshape"(%15) : (tensor<1xui32>) -> tensor<ui32>
    %17 = "stablehlo.add"(%arg6, %arg7) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %18 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %19 = "stablehlo.shift_left"(%arg7, %18) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %20 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %21 = "stablehlo.subtract"(%20, %16) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %23 = "stablehlo.shift_right_logical"(%arg7, %22) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %24 = "stablehlo.or"(%19, %23) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %25 = "stablehlo.xor"(%17, %24) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %26 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %27 = "stablehlo.reshape"(%26) : (tensor<1xui32>) -> tensor<ui32>
    %28 = "stablehlo.add"(%17, %25) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %29 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %30 = "stablehlo.shift_left"(%25, %29) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %31 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %32 = "stablehlo.subtract"(%31, %27) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %33 = "stablehlo.broadcast_in_dim"(%32) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %34 = "stablehlo.shift_right_logical"(%25, %33) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %35 = "stablehlo.or"(%30, %34) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %36 = "stablehlo.xor"(%28, %35) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %37 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %38 = "stablehlo.reshape"(%37) : (tensor<1xui32>) -> tensor<ui32>
    %39 = "stablehlo.add"(%28, %36) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %40 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %41 = "stablehlo.shift_left"(%36, %40) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %42 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %43 = "stablehlo.subtract"(%42, %38) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %45 = "stablehlo.shift_right_logical"(%36, %44) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %46 = "stablehlo.or"(%41, %45) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %47 = "stablehlo.xor"(%39, %46) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %48 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %49 = "stablehlo.reshape"(%48) : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.add"(%39, %47) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %51 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %52 = "stablehlo.shift_left"(%47, %51) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %53 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %54 = "stablehlo.subtract"(%53, %49) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %56 = "stablehlo.shift_right_logical"(%47, %55) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %57 = "stablehlo.or"(%52, %56) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %58 = "stablehlo.xor"(%50, %57) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %59 = "stablehlo.broadcast_in_dim"(%arg8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %60 = "stablehlo.add"(%50, %59) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %61 = "stablehlo.broadcast_in_dim"(%arg9) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %62 = "stablehlo.add"(%58, %61) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    %63 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %64 = "stablehlo.add"(%arg5, %63) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %65 = "stablehlo.convert"(%64) : (tensor<i64>) -> tensor<ui32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<3xui32>
    %67 = "stablehlo.add"(%62, %66) : (tensor<3xui32>, tensor<3xui32>) -> tensor<3xui32>
    "func.return"(%14, %60, %67, %arg9, %arg10, %arg8, %arg12, %arg11) : (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<5x4xbf16>) -> tensor<5xi64>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<5x4xbf16>):
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<5x4xi64>
    %1 = "stablehlo.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<i64>, %arg3: tensor<bf16>, %arg4: tensor<i64>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%11, %12) : (tensor<bf16>, tensor<i64>) -> ()
    }) : (tensor<5x4xbf16>, tensor<5x4xi64>, tensor<bf16>, tensor<i64>) -> (tensor<5xbf16>, tensor<5xi64>)
    "func.return"(%3#1) : (tensor<5xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

