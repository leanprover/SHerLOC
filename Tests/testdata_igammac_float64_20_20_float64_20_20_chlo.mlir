"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf64>, tensor<20x20xf64>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf64>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %9 = "stablehlo.compare"(%5#1, %8) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %10 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %13 = "stablehlo.or"(%9, %12) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %14 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %15 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %16 = "stablehlo.compare"(%5#1, %15) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %17 = "stablehlo.compare"(%5#1, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %18 = "stablehlo.or"(%16, %17) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %19 = "stablehlo.log"(%5#1) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %20 = "stablehlo.multiply"(%5#0, %19) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %21 = "stablehlo.subtract"(%20, %5#1) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %22 = "chlo.lgamma"(%5#0) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %23 = "stablehlo.subtract"(%21, %22) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %24 = "stablehlo.constant"() <{value = dense<1.7976931348623157E+308> : tensor<f64>}> : () -> tensor<f64>
    %25 = "stablehlo.log"(%24) : (tensor<f64>) -> tensor<f64>
    %26 = "stablehlo.negate"(%25) : (tensor<f64>) -> tensor<f64>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %28 = "stablehlo.compare"(%23, %27) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %29 = "stablehlo.or"(%13, %28) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %30 = "stablehlo.not"(%29) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %31 = "stablehlo.exponential"(%23) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %32 = "stablehlo.and"(%30, %18) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %33 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %34 = "stablehlo.broadcast_in_dim"(%33) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %35 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %37 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %38 = "stablehlo.broadcast_in_dim"(%37) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %39 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %40 = "stablehlo.broadcast_in_dim"(%39) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %41:7 = "stablehlo.while"(%32, %5#0, %34, %36, %5#1, %38, %40) ({
    ^bb0(%arg40: tensor<20x20xi1>, %arg41: tensor<20x20xf64>, %arg42: tensor<20x20xf64>, %arg43: tensor<20x20xf64>, %arg44: tensor<20x20xf64>, %arg45: tensor<20x20xf64>, %arg46: tensor<20x20xf64>):
      %222 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %223 = "stablehlo.reduce"(%arg40, %222) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %224 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%224) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%223) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg33: tensor<20x20xi1>, %arg34: tensor<20x20xf64>, %arg35: tensor<20x20xf64>, %arg36: tensor<20x20xf64>, %arg37: tensor<20x20xf64>, %arg38: tensor<20x20xf64>, %arg39: tensor<20x20xf64>):
      %198 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %199 = "stablehlo.broadcast_in_dim"(%198) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %200 = "stablehlo.add"(%arg34, %199) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %201 = "stablehlo.divide"(%arg37, %200) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %202 = "stablehlo.multiply"(%arg38, %201) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %203 = "stablehlo.multiply"(%arg35, %arg37) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %204 = "stablehlo.multiply"(%200, %200) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %205 = "stablehlo.divide"(%203, %204) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %206 = "stablehlo.subtract"(%202, %205) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %207 = "stablehlo.add"(%arg39, %206) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %208 = "stablehlo.divide"(%arg37, %200) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %209 = "stablehlo.multiply"(%arg35, %208) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %210 = "stablehlo.add"(%arg36, %209) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %211 = "stablehlo.divide"(%209, %210) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %212 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %213 = "stablehlo.broadcast_in_dim"(%212) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %214 = "stablehlo.compare"(%211, %213) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %215 = "stablehlo.and"(%arg33, %214) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %216 = "stablehlo.select"(%arg33, %200, %arg34) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %217 = "stablehlo.select"(%arg33, %209, %arg35) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %218 = "stablehlo.select"(%arg33, %210, %arg36) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %219 = "stablehlo.select"(%arg33, %arg37, %arg37) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %220 = "stablehlo.select"(%arg33, %206, %arg38) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %221 = "stablehlo.select"(%arg33, %207, %arg39) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      "stablehlo.return"(%215, %216, %217, %218, %219, %220, %221) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>)
    %42 = "stablehlo.multiply"(%41#3, %31) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %43 = "stablehlo.divide"(%42, %5#0) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %44 = "stablehlo.not"(%18) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %45 = "stablehlo.and"(%30, %44) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %46 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %48 = "stablehlo.subtract"(%47, %5#0) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %49 = "stablehlo.add"(%5#1, %48) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %50 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %52 = "stablehlo.add"(%49, %51) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %53 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %55 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %56 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %57 = "stablehlo.add"(%5#1, %56) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %58 = "stablehlo.multiply"(%52, %5#1) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %59 = "stablehlo.divide"(%57, %58) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %60 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %61 = "stablehlo.broadcast_in_dim"(%60) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %62 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %66 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %67 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %68 = "stablehlo.negate"(%5#1) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %69 = "stablehlo.multiply"(%59, %68) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %70 = "stablehlo.subtract"(%67, %69) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %71 = "stablehlo.divide"(%70, %58) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %72 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %73:15 = "stablehlo.while"(%45, %59, %61, %48, %52, %72, %57, %58, %54, %5#1, %63, %65, %67, %68, %71) ({
    ^bb0(%arg16: tensor<20x20xi1>, %arg17: tensor<20x20xf64>, %arg18: tensor<20x20xf64>, %arg19: tensor<20x20xf64>, %arg20: tensor<20x20xf64>, %arg21: tensor<f64>, %arg22: tensor<20x20xf64>, %arg23: tensor<20x20xf64>, %arg24: tensor<20x20xf64>, %arg25: tensor<20x20xf64>, %arg26: tensor<20x20xf64>, %arg27: tensor<20x20xf64>, %arg28: tensor<20x20xf64>, %arg29: tensor<20x20xf64>, %arg30: tensor<20x20xf64>):
      %192 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f64>}> : () -> tensor<f64>
      %193 = "stablehlo.compare"(%arg21, %192) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %194 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %195 = "stablehlo.reduce"(%arg16, %194) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg31: tensor<i1>, %arg32: tensor<i1>):
        %197 = "stablehlo.or"(%arg31, %arg32) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%197) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %196 = "stablehlo.and"(%193, %195) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%196) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf64>, %arg3: tensor<20x20xf64>, %arg4: tensor<20x20xf64>, %arg5: tensor<20x20xf64>, %arg6: tensor<f64>, %arg7: tensor<20x20xf64>, %arg8: tensor<20x20xf64>, %arg9: tensor<20x20xf64>, %arg10: tensor<20x20xf64>, %arg11: tensor<20x20xf64>, %arg12: tensor<20x20xf64>, %arg13: tensor<20x20xf64>, %arg14: tensor<20x20xf64>, %arg15: tensor<20x20xf64>):
      %88 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %89 = "stablehlo.add"(%arg6, %88) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %90 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %91 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %92 = "stablehlo.add"(%arg4, %91) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %93 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %95 = "stablehlo.add"(%arg5, %94) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %96 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %97 = "stablehlo.multiply"(%92, %96) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %98 = "stablehlo.multiply"(%arg7, %95) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %99 = "stablehlo.multiply"(%arg9, %97) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %100 = "stablehlo.subtract"(%98, %99) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %101 = "stablehlo.multiply"(%arg8, %95) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %102 = "stablehlo.multiply"(%arg10, %97) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %103 = "stablehlo.subtract"(%101, %102) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %104 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %105 = "stablehlo.broadcast_in_dim"(%104) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %106 = "stablehlo.compare"(%103, %105) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %107 = "stablehlo.divide"(%100, %103) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %108 = "stablehlo.subtract"(%arg2, %107) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %109 = "stablehlo.divide"(%108, %107) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %110 = "stablehlo.abs"(%109) : (tensor<20x20xf64>) -> tensor<20x20xf64>
      %111 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %112 = "stablehlo.broadcast_in_dim"(%111) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %113 = "stablehlo.select"(%106, %110, %112) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %114 = "stablehlo.select"(%106, %107, %arg2) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %115 = "stablehlo.multiply"(%arg13, %95) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %116 = "stablehlo.subtract"(%115, %arg7) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %117 = "stablehlo.multiply"(%arg11, %97) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %118 = "stablehlo.subtract"(%116, %117) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %119 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %120 = "stablehlo.multiply"(%arg9, %119) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %121 = "stablehlo.add"(%118, %120) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %122 = "stablehlo.multiply"(%arg14, %95) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %123 = "stablehlo.subtract"(%122, %arg8) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %124 = "stablehlo.multiply"(%arg12, %97) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %125 = "stablehlo.subtract"(%123, %124) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %126 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %127 = "stablehlo.multiply"(%arg10, %126) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %128 = "stablehlo.add"(%125, %127) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %129 = "stablehlo.multiply"(%114, %128) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %130 = "stablehlo.subtract"(%121, %129) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %131 = "stablehlo.divide"(%130, %103) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %132 = "stablehlo.select"(%106, %131, %arg15) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %133 = "stablehlo.subtract"(%132, %arg15) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %134 = "stablehlo.abs"(%133) : (tensor<20x20xf64>) -> tensor<20x20xf64>
      %135 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %136 = "stablehlo.broadcast_in_dim"(%135) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %137 = "stablehlo.select"(%106, %134, %136) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %138 = "stablehlo.abs"(%100) : (tensor<20x20xf64>) -> tensor<20x20xf64>
      %139 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %140 = "func.call"(%139) <{callee = @integer_pow}> : (tensor<f64>) -> tensor<f64>
      %141 = "stablehlo.broadcast_in_dim"(%140) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %142 = "stablehlo.compare"(%138, %141) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %143 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %144 = "stablehlo.broadcast_in_dim"(%143) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %145 = "stablehlo.multiply"(%arg7, %144) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %146 = "stablehlo.select"(%142, %145, %arg7) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %147 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %148 = "stablehlo.broadcast_in_dim"(%147) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %149 = "stablehlo.multiply"(%100, %148) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %150 = "stablehlo.select"(%142, %149, %100) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %151 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %152 = "stablehlo.broadcast_in_dim"(%151) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %153 = "stablehlo.multiply"(%arg8, %152) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %154 = "stablehlo.select"(%142, %153, %arg8) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %155 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %156 = "stablehlo.broadcast_in_dim"(%155) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %157 = "stablehlo.multiply"(%103, %156) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %158 = "stablehlo.select"(%142, %157, %103) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %159 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %160 = "stablehlo.broadcast_in_dim"(%159) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %161 = "stablehlo.multiply"(%arg13, %160) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %162 = "stablehlo.select"(%142, %161, %arg13) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %163 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %164 = "stablehlo.broadcast_in_dim"(%163) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %165 = "stablehlo.multiply"(%arg14, %164) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %166 = "stablehlo.select"(%142, %165, %arg14) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %167 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %168 = "stablehlo.broadcast_in_dim"(%167) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %169 = "stablehlo.multiply"(%121, %168) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %170 = "stablehlo.select"(%142, %169, %121) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %171 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %172 = "stablehlo.broadcast_in_dim"(%171) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %173 = "stablehlo.multiply"(%128, %172) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %174 = "stablehlo.select"(%142, %173, %128) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %175 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %176 = "stablehlo.broadcast_in_dim"(%175) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %177 = "stablehlo.compare"(%113, %176) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %178 = "stablehlo.and"(%arg1, %177) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %179 = "stablehlo.select"(%arg1, %114, %arg2) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %180 = "stablehlo.select"(%arg1, %113, %arg3) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %181 = "stablehlo.select"(%arg1, %92, %arg4) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %182 = "stablehlo.select"(%arg1, %95, %arg5) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %183 = "stablehlo.select"(%arg1, %150, %arg7) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %184 = "stablehlo.select"(%arg1, %158, %arg8) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %185 = "stablehlo.select"(%arg1, %146, %arg9) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %186 = "stablehlo.select"(%arg1, %154, %arg10) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %187 = "stablehlo.select"(%arg1, %162, %arg11) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %188 = "stablehlo.select"(%arg1, %166, %arg12) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %189 = "stablehlo.select"(%arg1, %170, %arg13) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %190 = "stablehlo.select"(%arg1, %174, %arg14) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %191 = "stablehlo.select"(%arg1, %132, %arg15) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      "stablehlo.return"(%178, %179, %180, %181, %182, %89, %183, %184, %185, %186, %187, %188, %189, %190, %191) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<f64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<f64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<f64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>)
    %74 = "stablehlo.multiply"(%73#1, %31) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %75 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %76 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %77 = "stablehlo.subtract"(%76, %43) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %78 = "stablehlo.select"(%18, %77, %74) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %79 = "stablehlo.constant"() <{value = dense<0x7FF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %80 = "stablehlo.broadcast_in_dim"(%79) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %81 = "stablehlo.compare"(%5#1, %80) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %82 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %84 = "stablehlo.select"(%81, %83, %78) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %85 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %86 = "stablehlo.broadcast_in_dim"(%85) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %87 = "stablehlo.select"(%13, %86, %84) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    "stablehlo.custom_call"(%87, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    "func.return"(%87) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf64>, tensor<20x20xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0xFF3ED80EF7A6C4BF04B84D212162F2BF2E30AB03EC95F9BF3C876C608775ECBFCF24F0E9E27709C084AB6DB186520240C3FD4D152EDDD8BF1AFD251A6FB9EFBF3372A2AB452717407FBE71CCD2CCF2BF2CF94C96616703C0DBADB1A89665E33F46AA2E4C4CB30840271A2A09DA2DF33F5C7340842F97F3BF8F93C81C07C2F4BFF3A43CBB81391240C9BD69CB48D91B40AAC7811B58B4FCBF32519F6B7969F7BF9FEBF4DE06990C40CE503CBFE3C716C08EDC2F5BA5490940ACA3B052478AF9BF35B40F8A41E914C0A499EC469360DBBF90EBFD73E8590CC0D40CD1FA12A5E5BF68804D8641C5DF3F5EC2FC6DAE9B06C0C6BEB8F7924701C071BA500BB658E6BFE23E2C83113D0940F80A9F1F042B0D408879144F767FED3F5739E2C3921FF1BF4D34BEF835871340A2FE2ADDFCABE8BFD48E46DCF7E00E402A83799C54C3F83FE629087428730B406418AA3906F1D7BF150BC6F6A5E3E53FF3B4041F5A2BE73FC4FED0B809FA05401A0BE96ED32703402F18F36D30D51340A47DEC0C918911C0BB97C4A95D3005C078AE7CD7D59AF53FFB57440DAD6808C0C494E2C15B991040861F8B188D19E63F10E0EF1F605CEE3F9C8CBE53F09313C04C92911FED971040D4F2117677A207C0FC6DFD5ABA2AF23F6D81E65041E4FABFEAA159CB0A49F3BF5CB4316372DAF3BF0760D870CD75F4BF0AA18E82374EFF3FD99A4496A7B3F5BF6C684B992A3F1640C4D1E4F5F635EBBFE371937A2373D13FB57565448A1EFFBFE6359D461FAFFCBF1274430337DD0BC0E6701A40C030EABFD9F4E0FFC47D07C0D1D349CB17861240C8DEEFAA548DE8BF55576D3ABC8AF53F50EFD7EC951001C00854FDEE6D5DDABF7A56D17CB405F5BF6E924C7D63A2933F233D02EF1980FDBF8F229F1ECDBF0B40EAF3F3C11CAF0640484DDF61DE0D0AC0C0D26CAF4F091540F2636F44F7280DC0793626A15F7AF13FEDB284D6F18504C0BEA0990732B601C0106B6F5A80C211C068B2BF51B64FDCBFAA18761382F8EE3FFA23DD8BB20716C0257A51E417E50EC0A89057763304124069D1A2674451E3BF1265E5CFEFCAE7BF6C9B111378240B401B2F6FB714F70CC0C0784D9815AA0640652DA5F25E8A16C0B517FF29FDF6FB3FB99A9CF585D5A4BFE1B1E24F38CCE7BF7ED86B23B63B1240AE63EFE152C8FFBF4263DD24DC1AE23F428376E48376F83F5C7C3E008D9EF93F5FCF33D4CFA4074054F73BFE351E0EC0E034F113B454074006B4EBAA5589C6BF9344941A492403400E9AD89D7AD00540643B8E404E4BF53FD0812D0D8B3D05C08E60DBF7D81FF1BFE6C531EEA6FE0C4080A574122F1E17C0B0A28C6A669BF2BF7E6A3A43B3EBFFBF602A6878B63C01C0DCE6982E766CB93FEA6987E478CF00C05F9E22DE0ED3D7BF6BEA2A960D2DD63F54493B97D0BBF33F00DE832B852507C06BED441E43FCE63FE0C3B5EBA09317C0A10884C690920540717619E05E9400C0F843D6E5CA14FC3F54E4309B2718B3BF4232FCE014A8E4BF5368629D55271240B2A0900BE9B6E9BFF5C89C013355EEBFBC6E246C213EFF3F1E74D7A2C25915407A0168425EB2FDBF52350CBBBFAD0AC0B392819853B1D13F8AC183C76ABF12C0E51CB5BCD5ACD23FA1CBD9893981E13FA2AB46B8075A0B4086A9E81DF9020EC0785BCA153712DCBFC6AF8F6691B5FA3F4930A857E48DD63F0DBA2FCFB7B10E40D256FEFB79D0F4BF0BD400114FF70CC0810F9A276A8905C05201DB8EDFA3D33FBCF25DACA24D16C025DF3F96BBE81240AE33842BBE3516C0BCC368F870F401C0C4A13BE30B16F3BF42404534B6230A408B07356E10EE16C0D2D5645C43FE00409344DF1C90ACB73FEA50EC65AF551AC05275A5D4EBADE9BF6CCCAFEF0087CBBFA29ADAA7C8FE1040CA734C643C3C10C0E8223024525A0BC0826C7D8167B3074024E62CB5040EFE3FD45A1733ED5DFDBF3B79EA29DC1206C01E870F909605F53F64ED9C37968F10C07E5E716AE2C6EBBFE6C50EABFCB811C088FAA158B10504C0B69018554D85F13FD59BC5D21C41D53F5257BFE218BFFA3F2667DC1156060940EFE10E8E0F2C0DC0328B95EE1497F8BFB8B6B33E9C3DDBBFECAFF005720705407F8EE06DA0D90DC0BEB686B369E30B405DDF059EACC6FE3F9A29E0F750400440BC67182D3A031840AF23DAAEE107004030FE8B98DA13D33FA4D6B4059EE01C40E7CC8B10680306C0765358506C7E09C0DAA55B837D3E12406AA34C401D1425C0B86417BF0FABD43FB2F4331D947E0B401A980ED682C1B3BF8BA7277B74E617407EA68A8FAA930140D4B814FF81280EC090C07E32DB4CECBF86F0EB74FCE4F93FF10339493E8216C05A978B279D8500C000C66AE81E851940B03AD2F5B02CE8BFA3F50E495FB7D6BFB45BE5ACAA560140EC7FE64690DE19C0D69088461A20F3BF49C05C4236350740BCF65B0C3523E6BFD7ADFA3AC90F01C075106E06281DDB3F785AD44F6E7B0AC098A9742C6C620040370B4D2B2C240FC09AA9703BCB18F03F0A53D3D61763E2BF98DEE9016E4511C00845B91BB9C90AC07ED84357B048134023FF07B32AABF53F7EBCE7D232AE10407CF7397FC523ECBFAA18402D7F0F07C045639A7FD0C3134099973BBA780D1740A1AC369F98C7FBBF76190F4730000F40AAE1CA729593E73F702BC075A981FD3FAF8E6E0240A7F4BF542E502E8A1909C08430472B746C0440BC919470DB2D1240AC54D8234D61CE3F2546C5147A76923F925629CF7B840640240E6906222C1A408CD6938D8B0EEC3F7B15C01D2B670A40F1B612456E541040196447B16B8FE73F00E6B359FA61FE3FB44FB5AEB491B1BFB4C376D4057500C0607C06F7FA4FF0BFAE137BAFCF6EF03F8A3395DC482C14C0B6DFEDCCED740CC040D19A0A12D4DC3FF95B6A61A6B805402CFE24CC74C6064014683DFAF5351940FECE3726450DD7BFDE97420A331B0940552482A0112101C04745D72EA672EEBF7708F9C86C371A405B876DC92EFF164044413A061D15F0BF8A7DEE686737FC3FED8CFDE84976E0BF30B8AC24C9851140168D699D23BD08C004884C70834EE03F1EF8EA179288DC3FE239C14B5B01F43F00531C7A789BF93F10A46FEE59000C40A8D237A4F44F1140BBEAD9A813CFE6BF1CDEADF8E4B9D8BF5C729184948D0A409EFBD0708148164002FC9C643C8306C0F6AB2E03ADC906C00054FDAF26B7FFBF274C5557518DFCBFDE339C17777CF53FCCFEF676B086F23FA6DB503F19DBED3FD35D380FC4D0F6BFAA551D33B3F2174098FF1FE810D701C0FEBBDC5538AB1640B554C891B907C4BF1F170A9AA5C100C03816B80CEC23FD3F5AE92647FB14FC3F52158809DCAB13C0355F625ECE1800C0604F25489EC50FC0AAF4EA8DCEAD15408040EBAF4F7F0FC0681613A62D92D0BF743BC814604EFEBFEC99720EF3610140BAF8309EAA8B0DC0294D9FB0F6A61AC0DA9C42BC094DF83FD85BDFEB515EFDBFC4F88AEE9AAB074037FB52358D4CEABF0EA0B5D050F300C0B0A6980D74270CC0606BD29DFE8F04C0689559F5687601C0C454DA96D1A20EC07F299AB33B5BE1BF63A2EA706F1E0540C0817BC9C73BFA3FF04F2BC3BEE1C83F434BC141AE55D43F7810320A8349F43F98783070C36EE7BF71A1F96DE0651140066F78EB668017403613903BD4C1D5BFF9BCABF70663FB3F6A95C9A990C0B93F087F8E7CCD6FE63F9AAACCA865E20040046F283B6440004008B73589BE4B02C08665CD00DF0000C03D5948E41C3E144076CF39111AEB10C04205D03F5D8BEA3FB42D6A46D3E70A4027DB95E712B913C0824689FDCA06C43FAA1FF20AB216F1BF48CF343AA4A8BE3FFA5DFCECEF4C1740762A2AC33D7C08C0E5B9BFEFB2120140F89051437DEEE7BFC3151D0670C912C04E7AEDB6E14B15C0A5D6C8A94753C2BFFCEE5078BE76F8BFA1C6228C3C83D6BF003833C456F6CABF354D4CD14C75CCBF6854D9E37B500AC07CAAE99EEBA2FA3FB433C0518A33D3BF98813C6CDD8212406218C95FEE9FFE3F355200F22B89C4BFF92815D6EA49B53FE40E21ED0487B9BF5040E58E91DEE2BF3896614865DDE3BF60BA50EED4BCEF3F64C82049A2DCCB3F12B2BCB2A23103C0902A190998110AC0EAF158379CEBD8BF68EA78B4CE997EBFDB4090AF4C9FF23FE5271311FDA0F53F9CD09944B2240EC09CC723ABCBF2F13FA8B010E264F3C23F9E19C021F12E11C0BE6F63F3B0CF02C0C0BA87E4D6791340143AE2EEA39400C03CC442ABCF010EC012596DAD6E190140DA2F301F5EB3EABF024E97C960C403C0FADE7D3204FB05C0EB4B5D551BB409C05E9D157DEB5F0B408A692BA6816BF6BFF9DA96A9356001C09C31007A499DF8BF6EDCF195D966084026ED1A06342CD3BF865D210DCA7809C0F0FA0F34D573FFBF9C2B49B90C13E73FD12BC6E519671040A4CBEFB5312EDC3F18169A222D1FFA3FEFF62EB8FEE309C0C61C1CB0021602401EC0B328E8F4FDBF8996CA86946FF83FFA1FA9CECF65FEBF"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    %4 = "stablehlo.constant"() <{value = dense<"0x42D69B95C92D0940A1AD3592BE80EC3FB00A08A33FF7F53FF853DF8E08D30640CAB05C5101FCE5BF690B1EFB9CB002C0C9B54A9976E8D53F66FA5910493E06C03C3D0B1B867ED13F9997FEFD0680FBBFF79CD8BCD076ECBF1332C294CEE6104031A687ED5CC114402ABECE61C0F914407D0F92BBDCF2F23FACB78BCD8B2E0140FE6E3F7A8FEEECBF1D802EBD2B8FF4BFF3914464FC1F00C01AD1483B50D50040B2E0889242AE06405558B37E1F9E06C048A4E73C5ACA05C0A426B92C0E59B0BFCA5B9A716F3AF6BFE0408C6B8794E1BF74029475BDE00CC0182090F20FA322C0D5EFF0B4489B11C0484333EF37962040ACF908D1C0971340F23B834D29200F40B72E40AFDC6CF53F0F43FCB8DF38F13F441AEC4C2CA40BC0378481060EA2D83F2959F19DFC73F43FC269C58780A00640EC6E1DA8E488EA3FC42DAA6DFFA7D13FB25A188E4330114010B2E395C574CFBFA48EEAB82C11034038DD4D2C078407C0A4751336653809C073E9E0561AD312C056202F8ABD1C2240B74E08C30F8905C0BA36DA69F225FCBFD0254D5E4306FD3F7AAD4E771918E33FB85D6C57FD86D6BFD8AE5B73DA68CEBFBE6132599D4DE8BF369887EF1F880640B20B40BB2D6DEE3F33A0F97F87FF07C0A9C05C047C2C06C09158B6AA8040164096064032EE380340EACE0AE5E874E83F56051237F43111C0D0F6EC84E1A2FABFAD73AE944198E9BF5C6E526D0C180840D751BD6B89E4114017B97F16BA4CFDBF8498B6FCE7FE0540C2C34E6EB07710C05E3767362F3DF0BFA45BD1631638F13F9D05A64CA1FFFD3F2EBDB8DD7770C5BFC2C92E37DB6B17C00DE6F5E8EDE10140BF54DE9054F5D03F10D202BCB780164030CB0A208159F1BFBC96E709B50C0D40B473F47D40C2EDBF1E96CE5665810F40E19F836E9F810640262F051968DAE9BF1CF55F8664600740B85DA609B2DDD13F2A803A97D26CD53FD0CEF129112708C0E5D4471BA2E801C0D27F4EE320931940D01149E8A57F1BC08D64AD9C8D3901C0E83AE09FD3DAEDBF2C0B7B335BB80340D4EB2B21AAAEE83F709F774BD4D006C0F663820DA947FC3F8AA0CBF43054F8BFC06ED328C581EC3F60D3480AAE5D02C0220E478FAD33E5BF8B1D06FA938502C074CABF44AD4EFD3F4440280226C912403E60AD6107C00DC0963E5DDCDD2AEEBFBEA6000FF7DDD13FF6B8C55F5BB2FA3F589B0951AE0C2040763DBA633D4AE4BF4ABC52F4BFC102409288C03C116413C03F146D81F2C7044064BE7C5FB1901640C97D6E0A234902C0788360F2CF14F2BF8AAFB37FF8F5FBBF2A3D10AD900008C02C07FB40520DFFBFECCF27983D6C17C0E036ABC35B810E4094C246F60FEE01400C562C0828DF0B40243C136989180740E7F8B6E4D91FFEBF34F2470BA65BEBBF6265CFEF80FBFEBF782A1DDAC7F8F2BFF8D1CD8C51810E402D570BA4C72D10C0EE33FE8268F2FEBFC2E0BEF70F5F01C0747AE5C351EDD23F7198FA9BB36C124000B285FE4F5A13C01A413D6A1B2FF13F2CAD9CC5E43AE93F94B8567DE005F93F505B7FB2003515C0D8BB840A547DF3BF1AD3A4D33D7510C0F5BFAB338E9D134050F15EADE6750240DC8B6620886E16C0C88A88526DBAFA3F0C7E5E21BEFD0440249BBAD333E4C0BF5B9C664A6F7A01C0B6E79A298620983F0402A1D7DEE9F53F4D77DBDAF22D0C405ECF8E5EECBD0E40FA765E31D7F0D13F9E6DE4938D5608404BE0FF3D4CB0DE3FC7E81544F52F1340FAC441367C7FD33F9E7C9D4D5B2D03C072B3EF4E0FC80540A6B5FE08FB64FC3FD671B9A5618A10C049DD4D321FA507C08909E452B04DCDBF302152018EA9134018844A8344AEFF3FEABE43193C7D02C0068F7341667DFF3F386DA7F8C241F4BF126E9191C66BFB3F48EE9EF4B60712C0228E5D839BABF63F18AC951726A4E0BFEA6A7507D375D33F01F87B27EFCAABBF73E6E4907E1407C0C096A8B7F54BE2BF8FADCBB05657E03F70D7694AC11B11401AAE91F556A4F5BFB87CF4BE5F1CF0BFB06DEB11E6600E406350658D748A15406CC99F511A4F03C0326F014D822508C0C56CC7C6A96A04406A5DFDA26E67FBBFB814D8B752B6F63F9DB71545E737F23F02A7C8EC508112401D787D68F0D50B406D8BD63C0CCBF7BF5BB843555ECD114098F85408CBD90DC08845D346DCD6E9BF9EDE36E24B13ED3F78657F75E83107C0510C3A82E30F05C08F06138933A10DC00E712C7E1ACBE63FF359534B8A180D407504597984CB0B4043EBF4A7E5830FC069563C1DFF7D004076132CF2F31AE73F453A9F102A6EDABFF64234DD847F0AC0C00CBD20FECFA23F09E787692D9200C0CCB6AB7249CE15C011A0D871A370024017445BD33307C63FFDCFB0CF9754D03F6414576A0BA10BC0BE74EA4B7F460940AC6466EC5A09DC3F6C98D7C453D4F3BF1276A3F71466F33F738DA54DF341F73F8E8EB08894E01240CD51FB314AC3F0BFE19908AE1F4AE33FD014A509B92418C0E19D5F43BA28064006B41101238A0FC0E0369A68CCF8104093D51A790CE30140303157FCB99E1B40B62E95A4DEE30340D1A9E829BBB61140DCC08634D52303C0FF7D04B76DEC06C05A93C81E77F9EF3F7A6BBAD3216302406AEB95ED242EF4BF4491852977ABDD3F4AFE7BE5302F0CC0D89D493A235D1AC03D94FB45CD850EC073F5A27A0CD206C032C9526393BF1240D7B1F5FDE93D0CC0EFD86941811901402D9DFA71560AFDBF6191D0AE3272F2BFBAD72BF33EFBF93F06BEB95EB754E33FC03408572FC7DE3F68D40A961E04BA3F4CEEADDE79A203408E0EDCEE1665F4BF54FDC10DB7570240F47C9EACFECBE9BF5B6162D157C800401A2C6414B4B214C0F2E77ABAB43CF9BFD2767614F6A0C73F0069B31B233DFB3F2A883930BC8DCABF551FD7C09E7EFB3FB2FF6685B6AEB53F1AEB0A80923F0E40A0CF4B512DCD15408AC3FE52FB76E3BF41297196724B08C04260E475A7ECF9BFCCD77DB2B4D61540912B6E5E5F8001C006F4C6744AA8C8BFF445EB25637A0340AB24C69E726B07C0885E2244BBF8F13FB631AA383FBAE3BFF635C182F1ECF9BF46D38E07B7B4ADBF08288C698F22F7BF1A80D581FE3BBD3F78EDCBF9A987E8BF247224C74FDAFDBFDE83046579DA0EC0D19A98A615B5E0BF617B9FBB8BDAF0BF73CEAB6A036203C0DB8B2C052A0FFCBFE60F0CEB81A0DBBFDC4106926830F33FBEC13D1A9F0D14C070F5C29C8270FDBF52DAD77343B50FC056BD4B61553F12C03908C6B73CC8014033F7605DFD58F53FF82C16CC719ED7BF4EB7E503AAAE0540F2B32B93F9690BC06FA9A1D7CFEFC23FD0FE12624B2BEBBF62CF63907EA60DC0BEF57000B264EFBF3EE6A5617C700CC099FE06509BA7F6BF197EB181D458E0BF961A98D6262914407C935FEC9668FE3F639EAF0DF1F9A23F74108BE6CABE02C0DAC513397B9B00C09793DD5BC3A51140C2B6BD95E9410840F0BC3C131959D6BF9FE5AF0D79060AC0F012E10EDD9BFC3FF6D59162177BEC3FA6F02018DAD3EB3F5632F5CB8651D33F68F2F0E397CA00C07B30C1F1FE3613C0F0995B07D4061340B159649C88B01240181CB92F5F2017C0EDB28A90B258E03FAEB7209E73F10A40FBCCED0F57DC0B406542F6982176F93F2E6A36E2C42E00C0B45F5BD7160DEFBF7A1D8727F2E211C01D1BAE2243C0F7BF13B0E4A4BD380A400CA0BD0F8EA5DC3FD3956BA0531406C001D926BEA2490240D39375EF4383FEBF9C6F6E7189671EC03E0CB94B40A6B83FA224349598FEECBF3E27C5FFE6C2E53F767892550B4BF7BF4AEDBB16F70709C04ED7630B052F0C40B40DA8BD180FFD3FB5A72C921106D6BFB1506E3F6F2BFCBF32D2AE725EE2F83F2E2F9B7FA2030640D4FA773DD666FEBF6EAAFE731132F5BF70F54DE2469FD93F4A205E395FDDFC3FCAA2C1F28DDD0A40F002EBE6ADCD0340EEE162F076E2014013A2EE16433B02C0497443DD80D3F83FF9B53D49406D15409D1EC133A4252040A2BCD3BE6A510E40826CFBA993EBFE3FB45400BD53DADE3FEFE5CB8C7BC2E7BF9F7C7CF4BE5C06C03D34047F9B6AFDBFFC6290BA622400C0D48568D4C039FF3F1D7A20DABECCEB3F66FF8176F1FDF83F4215946AC69804C08216A8D4EFC0054012499129F0BD1640791F719022DA02407F292B9DFF76F8BFD03B6CA0121608C0526839BA0CE219404AE164B593FB0240A18FA9606F8CF2BF4CDCF4C41C5B134097BF8124B14ED83FB806E2041BF2DA3FB07880C9C9A1094070F845DA8824EC3FAE226B8B932EE8BFA2F7266C3F8F014046F7AB9B140EEABF9C892FEFE72602C0F1D1C90CB0F1E0BF1888E7CF87480E40B95BA59B542510C08C5FB659E27BEABF2C8A6C15A9AFE23FE1C6520348EFF53FB3D13AFCF14D124020E463C04C79E03F6094E3BEEFD2FBBFB6665D2916D80440AD87FEC4310416C067A0E0CEED630340217071040ACDD0BF8E0D7AF6348701C00C29DF8B6B150840FF628BAD8AB503C091988C4E0979FB3F5049715EC10FE2BF"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%3, %4) : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F472B791FFEFFEF3F000000000000F03F000000000000F03F0F5C06BCD856753FF379152D116EBE3FBCE7E545CAE8803F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FA6343C00C10DE33F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F2B5BB36B7DE2EB3F3245207EB8B7EE3F000000000000F03F000000000000F03F4DA4A215919EEF3F000000000000F03F0F61DE244495EF3FDBF522E3D54FED3F60CC034F2053D13F000000000000F03FC7F1432D3B9DA83F000000000000F03F000000000000F03F000000000000F03FBDC896833B3DAA3F000000000000F03F000000000000F03F2AC4B374B49FD03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FF82CD1030F98EF3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FDF0ACD62B417EC3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F3768ECEB91DBC63F000000000000F03F000000000000F03F000000000000F03FFDA51FE1FBFC1E3F000000000000F03FE404E9850E97D53F401F9F09B245DB3F000000000000F03FF444D15E6C7CEB3F000000000000F03F9E9443D5CA2AE83F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F8D82626622E6EF3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F12275FE9D227E03F4231A6082385D63F2E0445B4257D563F000000000000F03F000000000000F03F000000000000F03F000000000000F03F2ECC7F33F2A5A43F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FE2BEF9D7B74A5D3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F32FEF4F50760A43F000000000000F03F000000000000F03FAE1E654798E5EF3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F0E8D60282111843F000000000000F03F000000000000F03F000000000000F03F000000000000F03FD9F891D90ADFB63FBEDAF61A1377693FC5FA566B59FDEF3F000000000000F03F000000000000F03F000000000000F03F8ACD7D580BB0D13F000000000000F03FAEAEE3C5DB6EEA3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FA67D5276F6ABDC3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F2F04043270DEEF3F000000000000F03F000000000000F03F000000000000F03FDED3A4A03480E73F000000000000F03F000000000000F03F000000000000F03F000000000000F03F6168E89E9E77773F000000000000F03F000000000000F03F7236521B30FAE13F000000000000F03F000000000000F03F000000000000F03F7560F17B100ABD3F000000000000F03F000000000000F03F38073E9C6476AD3F000000000000F03F000000000000F03FD894499290A6E83F000000000000F03F000000000000F03F000000000000F03F000000000000F03FE56BFC86D1D9E33F000000000000F03F000000000000F03FDDFE38022A1DE83F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FC65249D0FFFFEF3F000000000000F03F000000000000F03FD2D703C6664DEE3F000000000000F03F000000000000F03FC98A23EA55ABE93F000000000000F03F000000000000F03F1C5C20431B1FCD3F000000000000F03FC78E2E07429CCF3F000000000000F03FE5EEFB19F4D08D3F000000000000F03F000000000000F03F000000000000F03F300F30C32B63E03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F63A366AEDBFFEF3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F1170671295CCE03F000000000000F03F000000000000F03F63BC31C7D13E593FC5A615AA3BF4EE3FD0730459FAFFEF3FD8AA44E680B0EB3FD4D313ACD010E43F000000000000F03FE2907B083DB7AE3F000000000000F03F000000000000F03F000000000000F03F000000000000F03FDC3BF65FC2E9EA3F000000000000F03F000000000000F03F3A8FD880EA24AC3F26BABEF7C5FDEF3F12856E95DA19CF3F6CD60DA096CDE23F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F181A8A2BB63DEE3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F818C14B2FDA2B83F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F2C9DAF5F3849E03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FD84ED636D931E73F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F5AD4E3B29CB2EA3F53147560F5BD623FB8B49E5ABE6B703F5B67AB04EA69D23F000000000000F03F000000000000F03F000000000000F03F000000000000F03F8D8767E561ACBD3F80DF33A052D3B03F000000000000F03FB9DD62F94E4DD73F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FBE741066461C893F000000000000F03F000000000000F03F7CDD89B820C8EF3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F122534EDE47B583F000000000000F03F541E9179C7C2ED3FFC662CFA52DCEC3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03FA2DD5E362A9BDA3F9E75DC2C6F7A9A3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F8ABF3552E43BBD3F000000000000F03F000000000000F03F000000000000F03F51E7B6941BFFEF3F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F000000000000F03F2830FC57FDC4D63F000000000000F03F000000000000F03F000000000000F03FDD13C9C51328EB3F000000000000F03F000000000000F03F000000000000F03FFBFA0C8E7B1EA53F000000000000F03F2F9A5FB4984C973F000000000000F03F000000000000F03FE9F260F22215D03F000000000000F03F58A22A9BA7A2D53F000000000000F03F"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%2) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f64>) -> tensor<f64>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f64>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "func.return"(%1) : (tensor<f64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

