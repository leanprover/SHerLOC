"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf32>, tensor<20x20xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %7 = "stablehlo.compare"(%5#0, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %8 = "stablehlo.compare"(%5#1, %5#1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %9 = "stablehlo.or"(%7, %8) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %10 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %12 = "stablehlo.compare"(%5#1, %11) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %13 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %14 = "stablehlo.broadcast_in_dim"(%13) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %15 = "stablehlo.compare"(%5#1, %14) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %17 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %18 = "stablehlo.compare"(%5#1, %17) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %19 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %21 = "stablehlo.compare"(%5#0, %20) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %22 = "stablehlo.or"(%18, %21) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %23 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %24 = "stablehlo.broadcast_in_dim"(%23) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %25 = "stablehlo.compare"(%5#1, %24) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %26 = "stablehlo.compare"(%5#1, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %27 = "stablehlo.and"(%25, %26) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %28 = "stablehlo.log"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %29 = "stablehlo.multiply"(%5#0, %28) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %30 = "stablehlo.subtract"(%29, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %31 = "chlo.lgamma"(%5#0) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "stablehlo.subtract"(%30, %31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "stablehlo.constant"() <{value = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
    %34 = "stablehlo.log"(%33) : (tensor<f32>) -> tensor<f32>
    %35 = "stablehlo.negate"(%34) : (tensor<f32>) -> tensor<f32>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %37 = "stablehlo.compare"(%32, %36) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %38 = "stablehlo.exponential"(%32) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %39 = "stablehlo.or"(%12, %22) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %40 = "stablehlo.or"(%39, %37) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %41 = "stablehlo.or"(%40, %9) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %42 = "stablehlo.not"(%41) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %43 = "stablehlo.and"(%42, %27) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %44 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %45 = "stablehlo.broadcast_in_dim"(%44) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %46 = "stablehlo.subtract"(%45, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %47 = "stablehlo.add"(%5#1, %46) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %48 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %49 = "stablehlo.broadcast_in_dim"(%48) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %50 = "stablehlo.add"(%47, %49) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %51 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %52 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %53 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %55 = "stablehlo.add"(%5#1, %54) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %56 = "stablehlo.multiply"(%50, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %57 = "stablehlo.divide"(%55, %56) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %59 = "stablehlo.broadcast_in_dim"(%58) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %60 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %61 = "stablehlo.broadcast_in_dim"(%60) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %62 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %66 = "stablehlo.negate"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %67 = "stablehlo.multiply"(%57, %66) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %68 = "stablehlo.subtract"(%65, %67) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.divide"(%68, %56) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %71:15 = "stablehlo.while"(%43, %57, %59, %46, %50, %70, %55, %56, %52, %5#1, %61, %63, %65, %66, %69) ({
    ^bb0(%arg32: tensor<20x20xi1>, %arg33: tensor<20x20xf32>, %arg34: tensor<20x20xf32>, %arg35: tensor<20x20xf32>, %arg36: tensor<20x20xf32>, %arg37: tensor<f32>, %arg38: tensor<20x20xf32>, %arg39: tensor<20x20xf32>, %arg40: tensor<20x20xf32>, %arg41: tensor<20x20xf32>, %arg42: tensor<20x20xf32>, %arg43: tensor<20x20xf32>, %arg44: tensor<20x20xf32>, %arg45: tensor<20x20xf32>, %arg46: tensor<20x20xf32>):
      %231 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f32>}> : () -> tensor<f32>
      %232 = "stablehlo.compare"(%arg37, %231) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %233 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %234 = "stablehlo.reduce"(%arg32, %233) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %236 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%236) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %235 = "stablehlo.and"(%232, %234) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%235) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg17: tensor<20x20xi1>, %arg18: tensor<20x20xf32>, %arg19: tensor<20x20xf32>, %arg20: tensor<20x20xf32>, %arg21: tensor<20x20xf32>, %arg22: tensor<f32>, %arg23: tensor<20x20xf32>, %arg24: tensor<20x20xf32>, %arg25: tensor<20x20xf32>, %arg26: tensor<20x20xf32>, %arg27: tensor<20x20xf32>, %arg28: tensor<20x20xf32>, %arg29: tensor<20x20xf32>, %arg30: tensor<20x20xf32>, %arg31: tensor<20x20xf32>):
      %127 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %128 = "stablehlo.add"(%arg22, %127) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %129 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %130 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %131 = "stablehlo.add"(%arg20, %130) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %132 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %133 = "stablehlo.broadcast_in_dim"(%132) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %134 = "stablehlo.add"(%arg21, %133) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %135 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %136 = "stablehlo.multiply"(%131, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %137 = "stablehlo.multiply"(%arg23, %134) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %138 = "stablehlo.multiply"(%arg25, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %139 = "stablehlo.subtract"(%137, %138) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %140 = "stablehlo.multiply"(%arg24, %134) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %141 = "stablehlo.multiply"(%arg26, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %142 = "stablehlo.subtract"(%140, %141) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %143 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %144 = "stablehlo.broadcast_in_dim"(%143) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %145 = "stablehlo.compare"(%142, %144) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %146 = "stablehlo.divide"(%139, %142) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %147 = "stablehlo.subtract"(%arg18, %146) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %148 = "stablehlo.divide"(%147, %146) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %149 = "stablehlo.abs"(%148) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %150 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %151 = "stablehlo.broadcast_in_dim"(%150) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %152 = "stablehlo.select"(%145, %149, %151) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %153 = "stablehlo.select"(%145, %146, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %154 = "stablehlo.multiply"(%arg29, %134) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %155 = "stablehlo.subtract"(%154, %arg23) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %156 = "stablehlo.multiply"(%arg27, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %157 = "stablehlo.subtract"(%155, %156) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %158 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %159 = "stablehlo.multiply"(%arg25, %158) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %160 = "stablehlo.add"(%157, %159) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %161 = "stablehlo.multiply"(%arg30, %134) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %162 = "stablehlo.subtract"(%161, %arg24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %163 = "stablehlo.multiply"(%arg28, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %164 = "stablehlo.subtract"(%162, %163) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %165 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %166 = "stablehlo.multiply"(%arg26, %165) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %167 = "stablehlo.add"(%164, %166) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %168 = "stablehlo.multiply"(%153, %167) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %169 = "stablehlo.subtract"(%160, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %170 = "stablehlo.divide"(%169, %142) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %171 = "stablehlo.select"(%145, %170, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %172 = "stablehlo.subtract"(%171, %arg31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %173 = "stablehlo.abs"(%172) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %174 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %175 = "stablehlo.broadcast_in_dim"(%174) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %176 = "stablehlo.select"(%145, %173, %175) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %177 = "stablehlo.abs"(%139) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %178 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %179 = "func.call"(%178) <{callee = @integer_pow}> : (tensor<f32>) -> tensor<f32>
      %180 = "stablehlo.broadcast_in_dim"(%179) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %181 = "stablehlo.compare"(%177, %180) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %182 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %183 = "stablehlo.broadcast_in_dim"(%182) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %184 = "stablehlo.multiply"(%arg23, %183) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %185 = "stablehlo.select"(%181, %184, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %186 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %187 = "stablehlo.broadcast_in_dim"(%186) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %188 = "stablehlo.multiply"(%139, %187) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %189 = "stablehlo.select"(%181, %188, %139) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %190 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %191 = "stablehlo.broadcast_in_dim"(%190) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %192 = "stablehlo.multiply"(%arg24, %191) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %193 = "stablehlo.select"(%181, %192, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %194 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %195 = "stablehlo.broadcast_in_dim"(%194) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %196 = "stablehlo.multiply"(%142, %195) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %197 = "stablehlo.select"(%181, %196, %142) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %198 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %199 = "stablehlo.broadcast_in_dim"(%198) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %200 = "stablehlo.multiply"(%arg29, %199) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %201 = "stablehlo.select"(%181, %200, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %202 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %203 = "stablehlo.broadcast_in_dim"(%202) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %204 = "stablehlo.multiply"(%arg30, %203) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %205 = "stablehlo.select"(%181, %204, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %206 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %207 = "stablehlo.broadcast_in_dim"(%206) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %208 = "stablehlo.multiply"(%160, %207) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %209 = "stablehlo.select"(%181, %208, %160) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %210 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %211 = "stablehlo.broadcast_in_dim"(%210) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %212 = "stablehlo.multiply"(%167, %211) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %213 = "stablehlo.select"(%181, %212, %167) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %214 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %215 = "stablehlo.broadcast_in_dim"(%214) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %216 = "stablehlo.compare"(%152, %215) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %217 = "stablehlo.and"(%arg17, %216) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %218 = "stablehlo.select"(%arg17, %153, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %219 = "stablehlo.select"(%arg17, %152, %arg19) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %220 = "stablehlo.select"(%arg17, %131, %arg20) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %221 = "stablehlo.select"(%arg17, %134, %arg21) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %222 = "stablehlo.select"(%arg17, %189, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %223 = "stablehlo.select"(%arg17, %197, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %224 = "stablehlo.select"(%arg17, %185, %arg25) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %225 = "stablehlo.select"(%arg17, %193, %arg26) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %226 = "stablehlo.select"(%arg17, %201, %arg27) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %227 = "stablehlo.select"(%arg17, %205, %arg28) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %228 = "stablehlo.select"(%arg17, %209, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %229 = "stablehlo.select"(%arg17, %213, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %230 = "stablehlo.select"(%arg17, %171, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%217, %218, %219, %220, %221, %128, %222, %223, %224, %225, %226, %227, %228, %229, %230) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %72 = "stablehlo.multiply"(%71#1, %38) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %73 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %74 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %75 = "stablehlo.subtract"(%74, %72) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %76 = "stablehlo.not"(%27) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %77 = "stablehlo.and"(%42, %76) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %78 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %79 = "stablehlo.broadcast_in_dim"(%78) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %80 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %81 = "stablehlo.broadcast_in_dim"(%80) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %82 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %84 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %85 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %86:7 = "stablehlo.while"(%77, %5#0, %79, %81, %5#1, %83, %85) ({
    ^bb0(%arg8: tensor<20x20xi1>, %arg9: tensor<20x20xf32>, %arg10: tensor<20x20xf32>, %arg11: tensor<20x20xf32>, %arg12: tensor<20x20xf32>, %arg13: tensor<20x20xf32>, %arg14: tensor<20x20xf32>):
      %124 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %125 = "stablehlo.reduce"(%arg8, %124) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg15: tensor<i1>, %arg16: tensor<i1>):
        %126 = "stablehlo.or"(%arg15, %arg16) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%126) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%125) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf32>, %arg3: tensor<20x20xf32>, %arg4: tensor<20x20xf32>, %arg5: tensor<20x20xf32>, %arg6: tensor<20x20xf32>, %arg7: tensor<20x20xf32>):
      %100 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %101 = "stablehlo.broadcast_in_dim"(%100) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %102 = "stablehlo.add"(%arg2, %101) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %103 = "stablehlo.divide"(%arg5, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %104 = "stablehlo.multiply"(%arg6, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %105 = "stablehlo.multiply"(%arg3, %arg5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %106 = "stablehlo.multiply"(%102, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %107 = "stablehlo.divide"(%105, %106) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %108 = "stablehlo.subtract"(%104, %107) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %109 = "stablehlo.add"(%arg7, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %110 = "stablehlo.divide"(%arg5, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %111 = "stablehlo.multiply"(%arg3, %110) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %112 = "stablehlo.add"(%arg4, %111) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %113 = "stablehlo.divide"(%111, %112) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %114 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %115 = "stablehlo.broadcast_in_dim"(%114) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %116 = "stablehlo.compare"(%113, %115) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %117 = "stablehlo.and"(%arg1, %116) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %118 = "stablehlo.select"(%arg1, %102, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %119 = "stablehlo.select"(%arg1, %111, %arg3) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %120 = "stablehlo.select"(%arg1, %112, %arg4) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %121 = "stablehlo.select"(%arg1, %arg5, %arg5) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %122 = "stablehlo.select"(%arg1, %108, %arg6) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %123 = "stablehlo.select"(%arg1, %109, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%117, %118, %119, %120, %121, %122, %123) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %87 = "stablehlo.multiply"(%86#3, %38) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %88 = "stablehlo.divide"(%87, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %89 = "stablehlo.select"(%27, %75, %88) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %90 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %91 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %92 = "stablehlo.select"(%12, %91, %89) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %93 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %95 = "stablehlo.select"(%15, %94, %92) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %96 = "stablehlo.or"(%22, %9) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %97 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %98 = "stablehlo.broadcast_in_dim"(%97) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %99 = "stablehlo.select"(%96, %98, %95) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%99, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%99) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf32>, tensor<20x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0x3000773F967E22C0C0FDF4C0AAC3673F6A6C57BEBCB2B7BFA2B9EFBF572C00404C383BBD11D2FFBE2CCF73C07F80CEC010F650C0D2A84CC090CD2FC0532D463FEBB48EC0BD42DFBFF1B9AE40B511593F661C34C02B2A86BF44DBB83FA497873F40D4B73FF3526B4009B6A43FDA973A40ADC846C09665B5BF4A2B4B3FFC555040EF5CE13FCCB30CC0D1FDD23F4CAEC23FBA94223F398F55C00362BFBFA26C4440176B993F5B506E408B1F8C3F6A7976BF801B1240122FAD40283234408548E3409C2B9FC0878A974025B82BC0807A65C0A16C31BF49D4DA3F2A93993F3E9AE03FEB52B84074062EC01B8D2B4049351ABEA614043FCB0D664047547BC0D2CB13BF4D9D604033F936BFFA0A64C0E25036C0AE452A406E0F33401A63D73FBBEA5B40CDF1E4BFDE976FC07C387C407DAEA3408D3CBD3F74AF133FD4E5FB3FFC9B9F405E5492BE07C3BCBFF04DB43F6B6DBB3F5BA782BF1AF643BF9F2BB53FCA73A83F0427B940FAD77D3FD224C6BF553F8BBE715C67C03199CE3F36D65A3FF71C83BF41E0974087E1F6BFABC89E40952731404AF73EC0C7A2A2C0D80B5EBF5AEA42C0A2F2933F74390E40C4A41AC19F0999407A62AFC0CC67CC3FD6D0123FABFD164032C24CC0E4BAEC4051766A3EC6FFE53F1D3C853F5B1889BFCC4FB5BEEAA3673FA9E9554023966C3F818D8FBF656F113E6075193F20071ABF74C3923FAEE7EBBECC9C3840C08A3CC014856F4054BA1BBF32EDE43F87894240050B4ABFF392C840698CB9405EF8D8BF586F47405AC60BC057978E40264782C0D104584029DDE5C055A23C4070EF1AC030C3FF3FA6F4A2BF0DA394400FBDD63F4B0295C074CFF03E54FEBDC0BB6F2EBFEA7C4840B79B6F3EFDB508C0B7206BC060C71B3E48A5BD3F8C3B6DBFF607E4BE1758EB405FBF64C0CF794440774A95C0C2AE19404D33E340356489400D67BE3E7C33314091FB63C03AEC62C0E4EA2AC031E077403920624099211CC0DDF355C0E0E3F7BFF4272BC00532B23E48635F402F3B51403C499E40674B243FE3FC854018D59940FC17D7BF670B1140B7A3A53F7863203E44A68CBE1F2B3BC0685488BFA8010F40CAC3E4BFE012FB3F98F180BF0E3D3BC0BF2E893FD658173E1618BC3FE46BA1BFC7CEAB4044557E405112D440FDD1F7BF064A4ABFFEE007404EBCBD3E84E484BFB80C0BBF9104283F27A21CC0B520FB3F097E8540BD7E0640C4E87440101606C0FCBB7AC0152C29C061AB4FBFEE45AFBE92FFAA3F6957D2C0AE797EBFC7FE53C0143DADBE6F3E1EC09203B0BFDDC19AC0E4F08C40B655ABBF8E132840F30E8B40044664403E8113BF933230BF96DB544011D23CC0985ADDC0964B0140B2932EC01E1C913FBE203FC0062CFF3FB85D6F40384373401331903FC5CDABC047DDA5C052A8C13FC503A4BFF580624085C08D40EC787B3FFC11AEBFCCB56C40C8CA31C0AE3F00BD04AC0CBF8B0E0B40704CCB408062034158253FC003573F3F08309CBFFDC5A53FA86F7CC0DD8871BE7C9B0E41AD63C040DC516F403C809A40A5B7CBBEB4B65740DA27B73FDCA1FFC068341A40D23E453F21AF7C3E5F864BBEFC293D3FC94C09405F87F0BF455991BFC293F63E8D100740A126F1BEDA0C4F40C4268ABF718546403B1B1B4056D883BFBEC2C53F5F20AABFDAD94B3FBADB9D3F4323913F647E50401DF0583D56C24040559EA23E4441BBBF45855CC0AF409CBEF91F0FBEDA63A1C0AFEA2D4073C19340A1AAD7405E85B1C0746A02BF54AF8EBED0E46FBF07E76B403FD758BF057FAAC0076281C0B2029D40EC932CBF426E1140C74DD7BF49D0D33E7C95CA40B38536C084C1D73F28D7C7BFA51D3DBEA7302B40D1CC36BF76497EBF3626D63FE5079B3DBF0F8B4007AD45C004D90D401E04F9BE8E5CEE3FA9698840294F99C0B3A867C02A400840A1699B3FF3B622C036D2CD4000C51D3F4EC90E402FFC543FF5D5A6C0751FD440C5B1D03F643FE240926E0CC01AE68640E9220E3B54B098C0AA468DC029C23A403B799BC0453EBAC0D8999FC0B307B5405363D43FE7E026C0BFCE67C001D0C5BFB6F75340AF5B9540AD7B9E3F1BBB033ED216B4C0B31EC43EDFC3064024DA6040DE690DC09524FBC059C34DBFD11AA74065529EBF210124401E5C93C06409B03FC4A591BF6E0384BE55648CC0D3B787C0B15B2EC02150BD3F0172AB3F552880C0B7A661C05546323DC63E0D40C1265CBF5851CBBFFDF1DDBF16E3C03E131194C0D4310740"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    %4 = "stablehlo.constant"() <{value = dense<"0xB22DA8BEC7A123C0F9733C40CBAEF8C0FACB653E59EBB3BF497D08C075A02640642D78C0BB4802C003819B402ABC2940141C82BFA3C61A4037DD293E2412C9BF525685C0E503C04081BB5DBF07A90AC0C4432440240698C07FE4724057B3F13F869BA13E83E201406B849FC050AEE13F8BBBD0BFB6A9504010EB14BF09C16E3FF128F4BF116CC33FBE1D5B40501F2A40583A19C01C31C2C07E670C40774D60C0A1DFEDBFF69306C05B8C8DC040FA6B405926DCC07E3680BF368E88C038F733BFD30C173FA999D8BF86AEFCC0ECBEE5BF3DD321406C761DBF5361913F1C2D16400B6281BFDCAC3FBFB662564029A830402EF4533F5A440F41CE08D13FA84C0C40413434BFBDC140C06BB4463FC41D9DC0120BB94019B1E3BF0C3117C057433C4079C6E23F1A3281C0A68BF23E4B0C62C0614BD9C09F9B8E4011D2A8C042E92EC022F3DEBFB34302407C862B403C985540D28F41C00470A03F117350BFE8C2A1C0A417613F6F613EC00E7E303F91C6D6BF23A0CE3FBF5CBCBF46870C4063CF4DBFBAF725BF6A51B4400F2A893FED5A554082C7E8BF35AF3140415E5E4089A8F3BF2C1FCF3F06B6B240631FDCBF597483BFDE33AEBFBA0BED3FE055EDBF4BB4F93DE1FBF4405D3F08400506114030D08F40E6E4B9C04186094080F87CBE12A73940FBBBA440EB2328C0833E94C09B0E88BF3AF01EC0F34BE8BFE330B03F944E6BC0FB5E5D3FBA66DF4052A7B6C0A49609C0442D133FB8D7EABFF594D43FCD0676C0A11C953F02967940B5BC2FC08EDBB2BFE7E86140A059E9BFAC39DFBF34486AC01CB9E5BF0EBFE0BE7EF42A40F95D7CC0B9D9AA3FCD57F33EC30712414436614038C79DBEE1AB92BFBE8C81C0EC024340BE0F48405025EBBFB25CBF3FED33043FCDD5DBBFF04E1F4091E8693F050E76BD782E083F6C2984C0B5FBF13FDBE5843FA05A17C151683B3FE583C03EED7A2BC053A00940D8219E402192E73EB1614AC0A76011C022E0DEC015ACDC3FFBDC214084E4C340682046BFB7A669C03F502B3FCC007D403413A13F17A0C83F324037C0ED6F66BFEC125ABFFDE794C0323146C0D09719BEA5D2CD40533B8FC0758B4EC0AC2518408126C8BFDFB108C093AF40C0D3560BBF51B690C0BBB201C029109A3E976A9E3DCDECA140B8328AC00B93D7BE393D98C01311703F625895403C9196BF294B68C072BD863E3F77C83F8E6D8C3EF67F62C0BAB7483F480F773FE9688440AFD1B9C0B38441C00EAA833F7ABAB2BFBC1C2BC013A0FD3F3470643F3045273F2E59D9BFF273CABF01FDA3C0E9946CBFCC10CE3F034B403FA5F89C3FB298394077FD77BF8325F33F7193DE3ECDEBAEC02E9C07BFB01B633F560F8F3DF20E2B3F00611DBF36D79A3F99CA3A408D584F40962995409F9510C06D8783BFBD5F323F51EAB4BF52E6F1BF3EE95E40F34200C0853F0EBF21C472407BF2CF3F8D368D3FEED269C0777E32406480CCC0726513BFB58A98BFCC6960401825A2C0C33483BE3978923F86AB8F3E6FF284C08225913ED6570140241FE4BF52A354C0ED8397BFC6D3AABEB6451E3F4738FEBE9503F9BF35274A40CE2E19C0959B5D4019878740611FE7BF8AF659C0AC4978C0381E45403FAF38C0FB31603FCEAAA2401505E8BFE3B8B0BEC86B54C0F1AD8D40D604B63E60B54C3EE27E4C4050FF923F8686AABEAC832D3F4BE585C0E364A03F785301C0CB378B40D477B7402ED59BC090AB67BF1C5FB1C0ECB61CBF17C015BEB5E788BFC00C08403A72CCC0D8D6ABC0620C79C004EDC9C0A6A32A40BFB4814007D4F140CEE92FC0E2482BC0C15287C063F07840ACE467C0A6621BC0875B403E377CB9C0B8F9B440E019B53F2C2B81BFD3D7E3BECFABA73F902F6CC0E4DD853E8BBAA3C08D27AF3EB40EDC3EBFC324400D2965C01DA349C03E399F3F9E6FAFBFE1223CBE74DB933E4028AC3EEBFBD9BF9418F6BFF4542E40C10D743E10CBE8C020E57CC0F2E247C0093F45C03EB70EBF77F60540B47C9CBEF6B40E402FC1C240CA6E5EC0AA9CFA3FB35715C07CD9F73E2FD83DC06D052F3F50F2D1C0E1DB67C03E851840F828B240E21580C0189714BF5BA1F13F99E401C0D5CCA03FBC0046C0149501C02A15673F42E5B33FE86C6940C3DB823F245352400BCA103F17968FBF5A5597C0155DFEBF56C7ABBFA1E7CE40D4E635C07B2739BF00A71C4010BA4DC07A7FE93F5CDD243F29A4EEBFBB4569BFEECA453F7C51B9BEAE81463F100D9E3FD8F308C0"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%3, %4) : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F338F3B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F7E06733F2F8A553F72FEFF3DA3FF483E0000C07FCB518E3E0000C07F0000C07F0000C07F8DB6423D0000C07F0000C07FE0CE673F8B91583F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FEE3A193F79363E3F0000C07F0000C07F3E1A383F0000C07F345D4B3F3E727C3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0034733F0000C07F0000C07F4D11EC3E0000C07F0000C07F44E9D73A0000C07F0000C07FE10D7F3F0000C07F0000C07F0000C07F0000C07F428C5E3F65D16B3F0000C07F0000C07F0000C07F0000C07FB0B8F4390000C07F0000C07F0000C07F0000C07F0000C07FA1236A3F0000C07F0000C07F0000C07FED5BAA3BD49F323F0000C07F0000C07F0000C07F0000C07F8826413FFA58773F0000C07F0000C07F0000C07F17F32C3F0000C07FFF25133B0000C07FAA97733B5C307D3F9114743F0000C07F0000C07F0000C07FEB5A743FA10E593F0000C07F0000C07F0000C07F0000C07F0000C07FC323323F0000C07FCD5D8A3D0000C07F0000C07F0000C07FA21E203E0000C07F0000C07F0000C07F984FF03A0000C07F0000C07F0000C07F614DC13E0000C07F0000C07F0000C07F0000C07F0000C07FFB193F3F0000C07F7707A13C05E3103E0000C07FB5287E3F0000C07F0000C07F0000C07FC4E47E3F0000C07F0000C07F6E467B3FFD5B593E0000C07F0000C07FB608BB370000C07FD349723C0000C07F695BE43EFED1B3380000C07FFDA8563F99B9373C0000C07F0000C07F0000C07FDDD7D53A0000C07F0000C07F0000C07F0000C07F0000C07F3EF07F3F0000C07F0000C07F61EF3C3A9B017E3FD36EF83C5C6AE63C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F3554323F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F040DB536683BD1355E3E983E0000C07F0000C07F0000C07F4B91613F0000C07F0000C07F0000C07F0000C07F28BDF33EEC90F4380000C07F00B23C3C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F9543943DF7D2B63CC9EAD73E0000C07F0000C07F5EBDA03B0000C07F0000C07F41E65E3E0000C07FCF6FD93E0000C07FA886AF3E11C3C63EF3A7E53E84CD7C3F0000C07F0000C07F03B5933E0000C07F0000C07F78DCBE3E0000C07F0000C07FB88F163F0000C07F0000C07F0000C07FE96B3A3F0000C07F0000C07F0000C07F18B37B3F0000C07F0000C07F0000C07F0000C07F0000C07FF12C1235EEDB3C3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F57F57E3F0000C07FB0977B3F9AE1683F0000C07F0000C07F0000C07F0D854A3F0000C07F677C2A3D0000C07F0000C07F0000C07F0000C07F5A61773F0000C07F77008C3E53F86F3F641C213F0000C07F45837A3F0000C07FE8CF6F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F6D58613F0000C07F0000C07F0000C07F7E057F3F0000C07F0000C07FF5760E3D0000C07F0000C07FDB076F3E0000C07F0000C07FEDFEF73E0000C07FA6CA73380000C07FD071F43C0000C07F83A1423F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FEE1CDD340000C07F0000C07FE703743F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FDDF97F3F0000C07F0000C07FF85A723F0000C07F0000C07F0000C07F9665F0370000C07F0000C07F0000C07F0000C07FA763B43EB30A353F0000C07F0000C07F0000C07F0000C07FE00CA73E0000C07F0000C07F0000C07F0000C07F2771883E0000C07F91903B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE6855A3F0000C07F0000C07F28397B3F0000C07F0000C07F0000C07F0000C07FB3AF583F0000C07F0000C07F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%2) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

