"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x4xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %111 = "func.call"() <{callee = @expected}> : () -> tensor<5x4xf64>
    %112 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %113 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %114 = "stablehlo.shift_right_logical"(%112, %113) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %115 = "stablehlo.convert"(%114) : (tensor<i64>) -> tensor<ui32>
    %116 = "stablehlo.broadcast_in_dim"(%115) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %117 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %118 = "stablehlo.convert"(%117) : (tensor<ui32>) -> tensor<i64>
    %119 = "stablehlo.and"(%112, %118) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %120 = "stablehlo.convert"(%119) : (tensor<i64>) -> tensor<ui32>
    %121 = "stablehlo.broadcast_in_dim"(%120) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %122 = "stablehlo.concatenate"(%116, %121) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %123 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %124 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %125 = "func.call"(%122, %123, %124) <{callee = @_uniform}> : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<5x4xf64>
    "stablehlo.custom_call"(%125, %111) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x4xf64>, tensor<5x4xf64>) -> ()
    "func.return"(%125) : (tensor<5x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %110 = "stablehlo.constant"() <{value = dense<[[0.52286734638827848, 0.63809277992222024, 0.48479882418789, 0.76266020446279748], [0.67996636822643519, 0.44532535364606862, 0.75625579280848321, 0.76073724858951675], [0.32504364334015712, 0.58233053152090486, 0.88008197627653684, 0.56040213468002809], [0.96747217212282344, 0.49304563867921836, 0.72374622890728268, 0.95975933869077212], [0.55588321000681051, 0.049615688944020686, 0.48405548065598669, 0.79875184812853339]]> : tensor<5x4xf64>}> : () -> tensor<5x4xf64>
    "func.return"(%110) : (tensor<5x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<5x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg8: tensor<2xui32>, %arg9: tensor<f64>, %arg10: tensor<f64>):
    %55 = "stablehlo.convert"(%arg9) : (tensor<f64>) -> tensor<f64>
    %56 = "stablehlo.convert"(%arg10) : (tensor<f64>) -> tensor<f64>
    %57 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1x1xf64>
    %58 = "stablehlo.broadcast_in_dim"(%56) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1x1xf64>
    %59 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<40xui32>
    %60 = "stablehlo.slice"(%arg8) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %61 = "stablehlo.reshape"(%60) : (tensor<1xui32>) -> tensor<ui32>
    %62 = "stablehlo.slice"(%arg8) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %63 = "stablehlo.reshape"(%62) : (tensor<1xui32>) -> tensor<ui32>
    %64 = "stablehlo.slice"(%59) <{limit_indices = array<i64: 20>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %65 = "stablehlo.slice"(%59) <{limit_indices = array<i64: 40>, start_indices = array<i64: 20>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %66 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %67 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %68 = "stablehlo.xor"(%61, %63) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %69 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %70 = "stablehlo.xor"(%68, %69) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %71 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %72 = "stablehlo.add"(%64, %71) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %73 = "stablehlo.broadcast_in_dim"(%63) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %74 = "stablehlo.add"(%65, %73) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %75 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %76 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %77:9 = "stablehlo.while"(%76, %75, %72, %74, %63, %70, %61, %66, %67) ({
    ^bb0(%arg20: tensor<i64>, %arg21: tensor<i64>, %arg22: tensor<20xui32>, %arg23: tensor<20xui32>, %arg24: tensor<ui32>, %arg25: tensor<ui32>, %arg26: tensor<ui32>, %arg27: tensor<4xui32>, %arg28: tensor<4xui32>):
      %108 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %109 = "stablehlo.compare"(%arg20, %108) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%109) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg11: tensor<i64>, %arg12: tensor<i64>, %arg13: tensor<20xui32>, %arg14: tensor<20xui32>, %arg15: tensor<ui32>, %arg16: tensor<ui32>, %arg17: tensor<ui32>, %arg18: tensor<4xui32>, %arg19: tensor<4xui32>):
      %105:8 = "func.call"(%arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19) <{callee = @None}> : (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %106 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %107 = "stablehlo.add"(%arg11, %106) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%107, %105#0, %105#1, %105#2, %105#3, %105#4, %105#5, %105#6, %105#7) : (tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %78 = "stablehlo.concatenate"(%77#2, %77#3) <{dimension = 0 : i64}> : (tensor<20xui32>, tensor<20xui32>) -> tensor<40xui32>
    %79 = "stablehlo.slice"(%78) <{limit_indices = array<i64: 20>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %80 = "stablehlo.slice"(%78) <{limit_indices = array<i64: 40>, start_indices = array<i64: 20>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %81 = "stablehlo.convert"(%79) : (tensor<20xui32>) -> tensor<20xui64>
    %82 = "stablehlo.convert"(%80) : (tensor<20xui32>) -> tensor<20xui64>
    %83 = "stablehlo.constant"() <{value = dense<32> : tensor<ui64>}> : () -> tensor<ui64>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<20xui64>
    %85 = "stablehlo.shift_left"(%81, %84) : (tensor<20xui64>, tensor<20xui64>) -> tensor<20xui64>
    %86 = "stablehlo.or"(%85, %82) : (tensor<20xui64>, tensor<20xui64>) -> tensor<20xui64>
    %87 = "stablehlo.reshape"(%86) : (tensor<20xui64>) -> tensor<5x4xui64>
    %88 = "stablehlo.constant"() <{value = dense<12> : tensor<ui64>}> : () -> tensor<ui64>
    %89 = "stablehlo.broadcast_in_dim"(%88) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x4xui64>
    %90 = "stablehlo.shift_right_logical"(%87, %89) : (tensor<5x4xui64>, tensor<5x4xui64>) -> tensor<5x4xui64>
    %91 = "stablehlo.constant"() <{value = dense<4607182418800017408> : tensor<ui64>}> : () -> tensor<ui64>
    %92 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x4xui64>
    %93 = "stablehlo.or"(%90, %92) : (tensor<5x4xui64>, tensor<5x4xui64>) -> tensor<5x4xui64>
    %94 = "stablehlo.bitcast_convert"(%93) : (tensor<5x4xui64>) -> tensor<5x4xf64>
    %95 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %96 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<5x4xf64>
    %97 = "stablehlo.subtract"(%94, %96) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    %98 = "stablehlo.subtract"(%58, %57) : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %99 = "stablehlo.broadcast_in_dim"(%98) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %100 = "stablehlo.multiply"(%97, %99) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    %101 = "stablehlo.broadcast_in_dim"(%57) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %102 = "stablehlo.add"(%100, %101) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    %103 = "stablehlo.broadcast_in_dim"(%57) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %104 = "stablehlo.maximum"(%103, %102) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    "func.return"(%104) : (tensor<5x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<20xui32>, %arg2: tensor<20xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.add"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %3 = "stablehlo.reshape"(%2) : (tensor<1xui32>) -> tensor<ui32>
    %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %5 = "stablehlo.broadcast_in_dim"(%3) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %6 = "stablehlo.shift_left"(%arg2, %5) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %7 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %8 = "stablehlo.subtract"(%7, %3) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %10 = "stablehlo.shift_right_logical"(%arg2, %9) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %11 = "stablehlo.or"(%6, %10) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %12 = "stablehlo.xor"(%4, %11) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %13 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %14 = "stablehlo.reshape"(%13) : (tensor<1xui32>) -> tensor<ui32>
    %15 = "stablehlo.add"(%4, %12) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %16 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %17 = "stablehlo.shift_left"(%12, %16) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %18 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %19 = "stablehlo.subtract"(%18, %14) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %21 = "stablehlo.shift_right_logical"(%12, %20) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %22 = "stablehlo.or"(%17, %21) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %23 = "stablehlo.xor"(%15, %22) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %24 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %25 = "stablehlo.reshape"(%24) : (tensor<1xui32>) -> tensor<ui32>
    %26 = "stablehlo.add"(%15, %23) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %27 = "stablehlo.broadcast_in_dim"(%25) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %28 = "stablehlo.shift_left"(%23, %27) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %29 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %30 = "stablehlo.subtract"(%29, %25) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %31 = "stablehlo.broadcast_in_dim"(%30) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %32 = "stablehlo.shift_right_logical"(%23, %31) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %33 = "stablehlo.or"(%28, %32) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %34 = "stablehlo.xor"(%26, %33) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %35 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %36 = "stablehlo.reshape"(%35) : (tensor<1xui32>) -> tensor<ui32>
    %37 = "stablehlo.add"(%26, %34) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %38 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %39 = "stablehlo.shift_left"(%34, %38) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %40 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %41 = "stablehlo.subtract"(%40, %36) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %43 = "stablehlo.shift_right_logical"(%34, %42) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %44 = "stablehlo.or"(%39, %43) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %45 = "stablehlo.xor"(%37, %44) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %46 = "stablehlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %47 = "stablehlo.add"(%37, %46) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %48 = "stablehlo.broadcast_in_dim"(%arg4) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %49 = "stablehlo.add"(%45, %48) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %50 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %51 = "stablehlo.add"(%arg0, %50) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %52 = "stablehlo.convert"(%51) : (tensor<i64>) -> tensor<ui32>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %54 = "stablehlo.add"(%49, %53) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    "func.return"(%1, %47, %54, %arg4, %arg5, %arg3, %arg7, %arg6) : (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

